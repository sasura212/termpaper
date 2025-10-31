"""
Post-processing and Visualization for 400 kV Insulator Simulation

Features:
- Extract surface field distribution
- Generate publication-quality plots
- 3D visualization with PyVista
- Comparison with IEEE standards (Emax < 4 kV/mm for dry conditions)

Reference: Corona Ring Improvement to Surface Electric Field Stress Mitigation
           of 400 kV Composite Insulator (2024)
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    logger.warning("PyVista not available - 3D visualization disabled")

try:
    from dolfin import *
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    logger.warning("FEniCS not available - limited post-processing")


class ResultsPostProcessor:
    """Post-process and visualize electrostatic simulation results."""
    
    def __init__(self, results_dir="results", figures_dir="figures"):
        """
        Initialize post-processor.
        
        Args:
            results_dir: Directory containing solver output
            figures_dir: Directory for generated plots
        """
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # IEEE standards for insulator electric field
        self.E_critical_dry = 4.0e6  # 4 kV/mm (dry conditions)
        self.E_critical_wet = 2.5e6  # 2.5 kV/mm (wet/polluted)
        
    def load_results(self):
        """Load voltage and E-field from XDMF files."""
        if not FENICS_AVAILABLE:
            logger.error("FEniCS required to load results")
            return None, None
        
        voltage_path = self.results_dir / "voltage.xdmf"
        efield_path = self.results_dir / "e_field.xdmf"
        
        if not voltage_path.exists() or not efield_path.exists():
            logger.error(f"Results not found in {self.results_dir}")
            logger.info("Run solve_electrostatics.py first")
            return None, None
        
        logger.info("Loading simulation results...")
        
        try:
            # Load mesh
            mesh = Mesh()
            with XDMFFile(str(voltage_path)) as xdmf:
                xdmf.read(mesh)
            
            # Load voltage
            V_space = FunctionSpace(mesh, 'P', 2)
            V = Function(V_space)
            with XDMFFile(str(voltage_path)) as xdmf:
                xdmf.read_checkpoint(V, "voltage", 0)
            
            # Load E-field magnitude
            E_space = FunctionSpace(mesh, 'P', 1)
            E_mag = Function(E_space)
            with XDMFFile(str(efield_path)) as xdmf:
                xdmf.read_checkpoint(E_mag, "E_magnitude", 0)
            
            logger.info("  Results loaded successfully")
            
            return V, E_mag
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return None, None
    
    def extract_surface_profile(self, E_mag):
        """
        Extract electric field along insulator surface.
        
        Args:
            E_mag: Electric field magnitude function
            
        Returns:
            Arrays of (z_coord, E_values)
        """
        logger.info("Extracting surface field profile...")
        
        mesh = E_mag.function_space().mesh()
        coords = mesh.coordinates()
        
        # Find surface points (maximum radial distance at each z)
        z_unique = np.unique(coords[:, 2])
        z_surface = []
        E_surface = []
        
        for z in z_unique:
            # Points at this z-level
            mask = np.abs(coords[:, 2] - z) < 1e-6
            if not mask.any():
                continue
            
            # Find max radius point (shed edge)
            pts_at_z = coords[mask]
            radii = np.sqrt(pts_at_z[:, 0]**2 + pts_at_z[:, 1]**2)
            idx_max_r = radii.argmax()
            
            # Get field value at this point
            point = pts_at_z[idx_max_r]
            try:
                E_val = E_mag(point)
                z_surface.append(z)
                E_surface.append(E_val)
            except:
                pass
        
        logger.info(f"  Extracted {len(z_surface)} surface points")
        
        return np.array(z_surface), np.array(E_surface)
    
    def plot_surface_field(self, z_coords, E_values):
        """
        Plot electric field distribution along insulator surface.
        
        Args:
            z_coords: Axial coordinates (m)
            E_values: Electric field values (V/m)
        """
        logger.info("Generating surface field plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert to kV/m and mm
        E_kV_m = E_values / 1e3
        z_mm = z_coords * 1e3
        
        # Main plot
        ax.plot(z_mm, E_kV_m, 'b-', linewidth=2, label='Surface E-field')
        
        # Critical thresholds
        ax.axhline(self.E_critical_dry/1e3, color='r', linestyle='--', 
                   linewidth=1.5, label='Dry limit (4 kV/mm)')
        ax.axhline(self.E_critical_wet/1e3, color='orange', linestyle='--', 
                   linewidth=1.5, label='Wet limit (2.5 kV/mm)')
        
        # Formatting
        ax.set_xlabel('Axial Position (mm)', fontsize=12)
        ax.set_ylabel('Electric Field (kV/m)', fontsize=12)
        ax.set_title('Surface Electric Field Distribution - 400 kV Insulator', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Annotations
        E_max = E_kV_m.max()
        z_max = z_mm[E_kV_m.argmax()]
        ax.annotate(f'Peak: {E_max:.1f} kV/m\nat z={z_max:.0f} mm',
                   xy=(z_max, E_max), xytext=(z_max+200, E_max*0.9),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=10, color='red', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.figures_dir / "surface_field_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        
        plt.close()
    
    def plot_voltage_contour(self, V):
        """
        Plot voltage equipotential contours (2D slice).
        
        Args:
            V: Voltage function
        """
        logger.info("Generating voltage contour plot...")
        
        mesh = V.function_space().mesh()
        coords = mesh.coordinates()
        
        # Extract mid-plane slice (y ≈ 0)
        tol = 50.0  # mm
        mask = np.abs(coords[:, 1]) < tol
        
        x_slice = coords[mask, 0]
        z_slice = coords[mask, 2]
        V_slice = [V(coords[i]) for i in np.where(mask)[0]]
        
        # Create grid for contour plot
        from scipy.interpolate import griddata
        
        x_grid = np.linspace(x_slice.min(), x_slice.max(), 100)
        z_grid = np.linspace(z_slice.min(), z_slice.max(), 150)
        X, Z = np.meshgrid(x_grid, z_grid)
        
        V_grid = griddata((x_slice, z_slice), V_slice, (X, Z), method='cubic')
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 10))
        
        levels = np.linspace(0, V_slice.max(), 20)
        cs = ax.contourf(X*1e3, Z*1e3, V_grid/1e3, levels=levels/1e3, 
                        cmap='RdYlBu_r', extend='both')
        
        # Equipotential lines
        ax.contour(X*1e3, Z*1e3, V_grid/1e3, levels=10, colors='black', 
                  linewidths=0.5, alpha=0.4)
        
        # Colorbar
        cbar = plt.colorbar(cs, ax=ax)
        cbar.set_label('Voltage (kV)', fontsize=12)
        
        ax.set_xlabel('Radial Distance (mm)', fontsize=12)
        ax.set_ylabel('Axial Position (mm)', fontsize=12)
        ax.set_title('Equipotential Contours (Mid-Plane Slice)', 
                     fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "voltage_contours.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        
        plt.close()
    
    def visualize_3d(self, stl_path="geometry/insulator_model.stl"):
        """
        Create 3D visualization using PyVista.
        
        Args:
            stl_path: Path to STL geometry file
        """
        if not PYVISTA_AVAILABLE:
            logger.warning("PyVista not available - skipping 3D visualization")
            return
        
        logger.info("Generating 3D visualization...")
        
        # Check if STL exists
        if not os.path.exists(stl_path):
            logger.warning(f"STL not found: {stl_path}")
            return
        
        # Load geometry
        mesh = pv.read(stl_path)
        
        # Try to load E-field data
        efield_vtk = self.results_dir / "e_field.vtk"
        if efield_vtk.exists():
            field_data = pv.read(str(efield_vtk))
            # Map field data to geometry (if compatible)
            # This requires careful spatial interpolation
        
        # Create plotter
        pl = pv.Plotter(off_screen=True)
        
        # Add geometry
        pl.add_mesh(mesh, color='lightblue', opacity=0.6, 
                   label='Insulator')
        
        # Add coordinate axes
        pl.add_axes()
        
        # Camera position
        pl.camera_position = 'iso'
        
        # Title
        pl.add_text("400 kV Composite Insulator - 3D View", 
                   position='upper_edge', font_size=12)
        
        # Save screenshot
        output_path = self.figures_dir / "insulator_3d_view.png"
        pl.screenshot(str(output_path))
        logger.info(f"  Saved: {output_path}")
        
        pl.close()
    
    def generate_summary_table(self, z_coords, E_values):
        """
        Generate summary statistics table.
        
        Args:
            z_coords: Axial coordinates
            E_values: Electric field values
        """
        logger.info("Generating summary table...")
        
        # Statistics
        E_max = E_values.max()
        E_mean = E_values.mean()
        E_std = E_values.std()
        z_max = z_coords[E_values.argmax()]
        
        # Check against standards
        margin_dry = (self.E_critical_dry - E_max) / self.E_critical_dry * 100
        margin_wet = (self.E_critical_wet - E_max) / self.E_critical_wet * 100
        
        # Create table
        summary = f"""
╔═══════════════════════════════════════════════════════════════════╗
║       400 kV COMPOSITE INSULATOR - ELECTRIC FIELD SUMMARY          ║
╠═══════════════════════════════════════════════════════════════════╣
║ Field Statistics:                                                  ║
║   Maximum E-field:        {E_max/1e3:>8.2f} kV/m ({E_max/1e6:.3f} kV/mm)   ║
║   Mean E-field:           {E_mean/1e3:>8.2f} kV/m                        ║
║   Std. Deviation:         {E_std/1e3:>8.2f} kV/m                        ║
║   Location of Max:        z = {z_max*1e3:>6.1f} mm                       ║
╠═══════════════════════════════════════════════════════════════════╣
║ IEEE Standards Compliance:                                         ║
║   Dry Condition Limit:    {self.E_critical_dry/1e6:>8.1f} kV/mm (4000 kV/m)       ║
║   Safety Margin (Dry):    {margin_dry:>8.1f} %                            ║
║   Status:                 {'✓ PASS' if E_max < self.E_critical_dry else '✗ FAIL'}                        ║
║                                                                    ║
║   Wet Condition Limit:    {self.E_critical_wet/1e6:>8.1f} kV/mm (2500 kV/m)       ║
║   Safety Margin (Wet):    {margin_wet:>8.1f} %                            ║
║   Status:                 {'✓ PASS' if E_max < self.E_critical_wet else '✗ FAIL'}                        ║
╠═══════════════════════════════════════════════════════════════════╣
║ Recommendations:                                                   ║
"""
        
        if E_max < self.E_critical_wet:
            summary += "║   • Design is EXCELLENT - meets all criteria                       ║\n"
        elif E_max < self.E_critical_dry:
            summary += "║   • Design is GOOD for dry conditions only                         ║\n"
            summary += "║   • Consider corona ring optimization for wet conditions           ║\n"
        else:
            summary += "║   • Design NEEDS IMPROVEMENT - exceeds dry limit                   ║\n"
            summary += "║   • Optimize corona ring position and diameter                     ║\n"
            summary += "║   • Review insulator shed profile                                  ║\n"
        
        summary += "╚═══════════════════════════════════════════════════════════════════╝\n"
        
        print(summary)
        
        # Save to file
        output_path = self.results_dir / "field_summary_table.txt"
        with open(output_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"  Saved: {output_path}")
    
    def run_all(self):
        """Execute full post-processing pipeline."""
        logger.info("=" * 60)
        logger.info("Post-Processing 400 kV Insulator Simulation Results")
        logger.info("=" * 60)
        
        if not FENICS_AVAILABLE:
            logger.error("FEniCS required for post-processing")
            logger.info("Attempting to read summary file...")
            summary_file = self.results_dir / "solution_summary.txt"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    print(f.read())
            return
        
        # Load results
        V, E_mag = self.load_results()
        if V is None or E_mag is None:
            logger.error("Cannot proceed without results")
            return
        
        # Extract surface profile
        z_coords, E_values = self.extract_surface_profile(E_mag)
        
        # Generate plots
        self.plot_surface_field(z_coords, E_values)
        self.plot_voltage_contour(V)
        
        # 3D visualization
        self.visualize_3d()
        
        # Summary table
        self.generate_summary_table(z_coords, E_values)
        
        logger.info("=" * 60)
        logger.info("Post-processing complete!")
        logger.info(f"Figures saved to: {self.figures_dir}")
        logger.info("=" * 60)


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Post-process 400 kV insulator simulation results"
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Results directory"
    )
    parser.add_argument(
        "--figures",
        default="figures",
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ResultsPostProcessor(
        results_dir=args.results,
        figures_dir=args.figures
    )
    
    # Run
    try:
        processor.run_all()
        logger.info("\n✓ Success! Check figures/ directory for plots")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())