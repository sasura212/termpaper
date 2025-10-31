"""
Electrostatic Field Solver for 400 kV Composite Insulator

Solves Laplace equation: ∇·(ε∇V) = 0
with appropriate boundary conditions and material properties.

Reference: Corona Ring Improvement to Surface Electric Field Stress Mitigation
           of 400 kV Composite Insulator (2024)
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check FEniCS availability
try:
    from dolfin import *
    FENICS_AVAILABLE = True
    logger.info("FEniCS detected - using full FEM solver")
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS not available - using analytical approximation")
    logger.warning("Install FEniCS for accurate results: conda install -c conda-forge fenics")


class ElectrostaticSolver:
    """Solve electrostatic field distribution in composite insulator."""
    
    def __init__(self, mesh_path="geometry/insulator_mesh.xdmf", 
                 output_dir="results"):
        """
        Initialize solver.
        
        Args:
            mesh_path: Path to XDMF mesh file
            output_dir: Directory for solution output files
        """
        self.mesh_path = Path(mesh_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Material properties (from paper Table 1)
        self.epsilon_r = {
            'air': 1.0,
            'silicone_rubber': 3.2,  # Housing material
            'frp_core': 5.5,          # Fiber-reinforced plastic
            'pollution': 4.5,         # Contaminated surface layer
            'corona_ring': 1.0        # Perfect conductor (enforced via BC)
        }
        
        # Voltage parameters (400 kV line-to-ground)
        self.V_HV = 400e3 / np.sqrt(3)  # Phase voltage in Volts (230.9 kV)
        self.V_GND = 0.0
        
        # Physical constants
        self.epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        
    def load_mesh(self):
        """Load mesh from XDMF file."""
        if not FENICS_AVAILABLE:
            logger.error("FEniCS required for mesh loading")
            return None
        
        if not self.mesh_path.exists():
            logger.error(f"Mesh file not found: {self.mesh_path}")
            logger.info("Run mesh_utils.py first to generate mesh")
            return None
        
        logger.info(f"Loading mesh: {self.mesh_path}")
        
        try:
            # Load mesh
            mesh = Mesh()
            with XDMFFile(str(self.mesh_path)) as xdmf:
                xdmf.read(mesh)
            
            # Load material markers (subdomains)
            mvc = MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with XDMFFile(str(self.mesh_path)) as xdmf:
                xdmf.read(mvc, "material")
            
            materials = cpp.mesh.MeshFunctionSizet(mesh, mvc)
            
            logger.info(f"  Mesh loaded: {mesh.num_vertices()} vertices, "
                       f"{mesh.num_cells()} cells")
            
            return mesh, materials
            
        except Exception as e:
            logger.error(f"Failed to load mesh: {e}")
            return None, None
    
    def define_boundaries(self, mesh):
        """
        Define boundary conditions.
        
        Args:
            mesh: FEniCS mesh object
            
        Returns:
            Dictionary of boundary markers
        """
        logger.info("Defining boundary conditions...")
        
        # Tolerance for boundary detection
        tol = 1e-3
        
        # Get mesh coordinates
        coords = mesh.coordinates()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        
        # Define boundary subdomains
        class HVBoundary(SubDomain):
            """High voltage terminal (top)."""
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], z_max, tol)
        
        class GroundBoundary(SubDomain):
            """Grounded terminal (bottom)."""
            def inside(self, x, on_boundary):
                return on_boundary and near(x[2], z_min, tol)
        
        class OuterBoundary(SubDomain):
            """Far-field boundary (Neumann: E·n = 0)."""
            def inside(self, x, on_boundary):
                r = np.sqrt(x[0]**2 + x[1]**2)
                r_max = coords[:, :2].max()
                return on_boundary and near(r, r_max, tol)
        
        # Mark boundaries
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundaries.set_all(0)
        
        hv_bnd = HVBoundary()
        hv_bnd.mark(boundaries, 1)
        
        gnd_bnd = GroundBoundary()
        gnd_bnd.mark(boundaries, 2)
        
        outer_bnd = OuterBoundary()
        outer_bnd.mark(boundaries, 3)
        
        logger.info(f"  HV boundary nodes: {np.sum(boundaries.array() == 1)}")
        logger.info(f"  Ground boundary nodes: {np.sum(boundaries.array() == 2)}")
        
        return boundaries
    
    def solve_fenics(self, mesh, materials, boundaries):
        """
        Solve Poisson equation using FEniCS FEM.
        
        Args:
            mesh: FEniCS mesh
            materials: Material subdomain markers
            boundaries: Boundary markers
            
        Returns:
            Voltage solution function
        """
        logger.info("Setting up FEM problem...")
        
        # Function space (P2 elements for better field accuracy)
        V = FunctionSpace(mesh, 'P', 2)
        
        # Trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)
        
        # Material property function (piecewise constant)
        class PermittivityExpression(UserExpression):
            def __init__(self, materials, epsilon_map, **kwargs):
                super().__init__(**kwargs)
                self.materials = materials
                self.epsilon_map = epsilon_map
            
            def eval_cell(self, values, x, cell):
                mat_id = self.materials[cell.index]
                # Map physical ID to material (default to air)
                if mat_id == 2:
                    values[0] = self.epsilon_map['silicone_rubber']
                elif mat_id == 3:
                    values[0] = self.epsilon_map['frp_core']
                elif mat_id == 7:
                    values[0] = self.epsilon_map['pollution']
                else:
                    values[0] = self.epsilon_map['air']
            
            def value_shape(self):
                return ()
        
        epsilon_r = PermittivityExpression(materials, self.epsilon_r, degree=0)
        
        # Weak form: ∫(ε∇u·∇v)dx = 0
        a = epsilon_r * inner(grad(u), grad(v)) * dx
        L = Constant(0.0) * v * dx
        
        # Boundary conditions
        bc_hv = DirichletBC(V, Constant(self.V_HV), boundaries, 1)
        bc_gnd = DirichletBC(V, Constant(self.V_GND), boundaries, 2)
        bcs = [bc_hv, bc_gnd]
        
        # Solve linear system
        logger.info("Solving linear system...")
        u_h = Function(V)
        
        solve(a == L, u_h, bcs,
              solver_parameters={
                  'linear_solver': 'mumps',  # Direct solver (robust)
                  'preconditioner': 'default'
              })
        
        logger.info("  Solution converged")
        
        return u_h
    
    def compute_electric_field(self, V_solution):
        """
        Compute electric field E = -∇V.
        
        Args:
            V_solution: Voltage solution function
            
        Returns:
            Electric field function (vector)
        """
        logger.info("Computing electric field...")
        
        # Vector function space for E-field
        mesh = V_solution.function_space().mesh()
        V_vec = VectorFunctionSpace(mesh, 'P', 1)
        
        # E = -grad(V)
        E = project(-grad(V_solution), V_vec)
        
        logger.info("  Electric field computed")
        
        return E
    
    def save_results(self, V_solution, E_field):
        """
        Save solution to XDMF for visualization.
        
        Args:
            V_solution: Voltage function
            E_field: Electric field function
        """
        logger.info("Saving results...")
        
        # Voltage
        V_file = XDMFFile(str(self.output_dir / "voltage.xdmf"))
        V_file.parameters["flush_output"] = True
        V_file.parameters["functions_share_mesh"] = True
        V_file.write(V_solution, 0.0)
        logger.info(f"  Saved: {self.output_dir / 'voltage.xdmf'}")
        
        # Electric field magnitude
        mesh = V_solution.function_space().mesh()
        V_scalar = FunctionSpace(mesh, 'P', 1)
        E_mag = project(sqrt(inner(E_field, E_field)), V_scalar)
        E_mag.rename("E_magnitude", "Electric field magnitude")
        
        E_file = XDMFFile(str(self.output_dir / "e_field.xdmf"))
        E_file.parameters["flush_output"] = True
        E_file.write(E_mag, 0.0)
        logger.info(f"  Saved: {self.output_dir / 'e_field.xdmf'}")
        
        # Extract statistics
        E_values = E_mag.vector().get_local()
        logger.info(f"\n  Electric Field Statistics:")
        logger.info(f"    Max: {E_values.max()/1e3:.2f} kV/m")
        logger.info(f"    Mean: {E_values.mean()/1e3:.2f} kV/m")
        logger.info(f"    Std: {E_values.std()/1e3:.2f} kV/m")
        
        # Save summary to file
        summary_path = self.output_dir / "solution_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("400 kV Composite Insulator Electrostatic Simulation\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Applied Voltage (HV): {self.V_HV/1e3:.1f} kV\n")
            f.write(f"Ground Voltage: {self.V_GND} V\n\n")
            f.write("Electric Field Statistics:\n")
            f.write(f"  Maximum: {E_values.max()/1e3:.2f} kV/m\n")
            f.write(f"  Mean: {E_values.mean()/1e3:.2f} kV/m\n")
            f.write(f"  Std Dev: {E_values.std()/1e3:.2f} kV/m\n")
        
        logger.info(f"  Summary: {summary_path}")
        
        return E_values.max()
    
    def analytical_approximation(self):
        """
        Fallback analytical solution for simplified geometry.
        Used when FEniCS is not available.
        """
        logger.info("Using analytical approximation (simplified model)...")
        
        # Assume cylindrical geometry: E_r = V / (r * ln(R_outer/R_inner))
        # This is VERY approximate - results NOT suitable for publication
        
        L = 3.5  # Insulator length (m)
        R_core = 0.04  # Core radius (m)
        R_shed = 0.16  # Shed radius (m)
        
        # Radial field at shed surface (uniform field approximation)
        E_radial = self.V_HV / (R_shed * np.log(R_shed / R_core))
        
        # Axial field (uniform along length)
        E_axial = self.V_HV / L
        
        E_max = np.sqrt(E_radial**2 + E_axial**2)
        
        logger.info(f"\n  Analytical Approximation Results:")
        logger.info(f"    E_radial: {E_radial/1e3:.2f} kV/m")
        logger.info(f"    E_axial: {E_axial/1e3:.2f} kV/m")
        logger.info(f"    E_max (approx): {E_max/1e3:.2f} kV/m")
        logger.warning("  NOTE: This is a rough estimate. Use FEniCS for accurate results.")
        
        # Save dummy output
        summary_path = self.output_dir / "solution_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("400 kV Insulator - ANALYTICAL APPROXIMATION\n")
            f.write("=" * 60 + "\n")
            f.write("WARNING: FEniCS not installed - using simplified model\n")
            f.write("Install FEniCS for accurate FEM results\n\n")
            f.write(f"E_max (estimate): {E_max/1e3:.2f} kV/m\n")
        
        return E_max
    
    def run(self):
        """Execute full solution pipeline."""
        logger.info("=" * 60)
        logger.info("Electrostatic Field Solver for 400 kV Insulator")
        logger.info("=" * 60)
        
        if not FENICS_AVAILABLE:
            return self.analytical_approximation()
        
        # Load mesh
        mesh, materials = self.load_mesh()
        if mesh is None:
            logger.error("Cannot proceed without mesh")
            return None
        
        # Define boundaries
        boundaries = self.define_boundaries(mesh)
        
        # Solve
        V_solution = self.solve_fenics(mesh, materials, boundaries)
        
        # Compute E-field
        E_field = self.compute_electric_field(V_solution)
        
        # Save results
        E_max = self.save_results(V_solution, E_field)
        
        logger.info("=" * 60)
        logger.info("Solution complete!")
        logger.info("=" * 60)
        
        return E_max


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Solve electrostatic field for 400 kV insulator"
    )
    parser.add_argument(
        "--mesh",
        default="geometry/insulator_mesh.xdmf",
        help="Path to mesh XDMF file"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--voltage",
        type=float,
        default=400.0,
        help="Line voltage in kV (default: 400)"
    )
    
    args = parser.parse_args()
    
    # Initialize solver
    solver = ElectrostaticSolver(
        mesh_path=args.mesh,
        output_dir=args.output
    )
    
    # Override voltage if specified
    solver.V_HV = args.voltage * 1e3 / np.sqrt(3)
    
    # Run
    try:
        E_max = solver.run()
        if E_max:
            logger.info(f"\n✓ Success! Maximum E-field: {E_max/1e3:.2f} kV/m")
            return 0
        else:
            logger.error("\n✗ Solution failed")
            return 1
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())