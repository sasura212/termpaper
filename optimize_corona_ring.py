"""
Corona Ring Parameter Optimization for 400 kV Composite Insulator

Objective: Minimize maximum surface electric field
Variables: Ring diameter, tube diameter, vertical position

Uses SciPy optimization with FEM solver in the loop.

Reference: Corona Ring Improvement to Surface Electric Field Stress Mitigation
           of 400 kV Composite Insulator (2024)
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import time
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoronaRingOptimizer:
    """Optimize corona ring parameters to minimize surface E-field."""
    
    def __init__(self, output_dir="optimization"):
        """
        Initialize optimizer.
        
        Args:
            output_dir: Directory for optimization results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Optimization bounds (based on paper Section 4)
        # All dimensions in mm
        self.bounds = {
            'ring_diameter': (520, 620),      # Total diameter of ring
            'tube_diameter': (30, 50),        # Diameter of tube cross-section
            'position_z': (50, 200),          # Distance from HV terminal
        }
        
        # History tracking
        self.iteration_history = []
        self.eval_count = 0
        self.best_result = {'params': None, 'E_max': np.inf}
        
    def modify_geometry(self, ring_diameter, tube_diameter, position_z):
        """
        Modify STL geometry with new corona ring parameters.
        
        This is a placeholder - in production, would use:
        1. CAD Python API (FreeCAD, CadQuery) to rebuild geometry
        2. STL manipulation libraries (trimesh, pymesh)
        3. Parametric Gmsh scripts
        
        Args:
            ring_diameter: Outer diameter of corona ring (mm)
            tube_diameter: Tube cross-section diameter (mm)
            position_z: Vertical distance from HV terminal (mm)
        """
        logger.debug(f"  Modifying geometry: D={ring_diameter:.1f}, "
                    f"d={tube_diameter:.1f}, z={position_z:.1f}")
        
        # PLACEHOLDER: In real implementation, regenerate STL here
        # For now, we'll use parametric scaling of existing geometry
        
        # Example using trimesh (if geometry manipulation is needed):
        # import trimesh
        # mesh = trimesh.load('geometry/insulator_model.stl')
        # # Apply transformations
        # mesh.apply_scale([scale_x, scale_y, scale_z])
        # mesh.export('geometry/insulator_modified.stl')
        
        pass
    
    def run_simulation(self, params):
        """
        Run FEM simulation with given parameters.
        
        Args:
            params: Array [ring_diameter, tube_diameter, position_z]
            
        Returns:
            Maximum surface electric field (V/m)
        """
        ring_diameter, tube_diameter, position_z = params
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation #{self.eval_count + 1}")
        logger.info(f"  Ring Diameter: {ring_diameter:.1f} mm")
        logger.info(f"  Tube Diameter: {tube_diameter:.1f} mm")
        logger.info(f"  Position: {position_z:.1f} mm from HV")
        logger.info(f"{'='*60}")
        
        # Modify geometry
        self.modify_geometry(ring_diameter, tube_diameter, position_z)
        
        # Run mesh generation (in practice, only if geometry changed)
        # os.system("python mesh_utils.py --stl geometry/insulator_modified.stl")
        
        # Run solver
        try:
            # Import solver
            sys.path.insert(0, os.path.dirname(__file__))
            from solve_electrostatics import ElectrostaticSolver
            
            solver = ElectrostaticSolver(
                mesh_path="geometry/insulator_mesh.xdmf",
                output_dir=self.output_dir / f"iter_{self.eval_count:03d}"
            )
            
            E_max = solver.run()
            
            if E_max is None:
                logger.warning("  Solver failed - returning penalty")
                E_max = 1e9  # Large penalty value
            
        except Exception as e:
            logger.error(f"  Simulation failed: {e}")
            E_max = 1e9
        
        # Update history
        self.eval_count += 1
        self.iteration_history.append({
            'iteration': self.eval_count,
            'ring_diameter': ring_diameter,
            'tube_diameter': tube_diameter,
            'position_z': position_z,
            'E_max': E_max
        })
        
        # Update best result
        if E_max < self.best_result['E_max']:
            self.best_result = {
                'params': params.copy(),
                'E_max': E_max
            }
            logger.info(f"  ★ NEW BEST: E_max = {E_max/1e3:.2f} kV/m")
        
        return E_max
    
    def objective_function(self, params):
        """
        Objective function for optimization.
        
        Args:
            params: Array [ring_diameter, tube_diameter, position_z]
            
        Returns:
            Objective value (E_max to minimize)
        """
        E_max = self.run_simulation(params)
        
        # Add constraint penalties if needed
        penalty = 0.0
        
        # Example: Penalize if ring is too close to insulator surface
        # (geometric constraint - 50mm minimum clearance)
        min_clearance = 50.0  # mm
        if params[2] < min_clearance:
            penalty = 1e6 * (min_clearance - params[2])**2
        
        return E_max + penalty
    
    def optimize_local(self, initial_guess=None):
        """
        Local optimization using Nelder-Mead or COBYLA.
        Fast but may find local minima.
        
        Args:
            initial_guess: Starting point [D, d, z] (mm)
            
        Returns:
            Optimization result object
        """
        if initial_guess is None:
            # Start from middle of bounds
            initial_guess = np.array([
                np.mean(self.bounds['ring_diameter']),
                np.mean(self.bounds['tube_diameter']),
                np.mean(self.bounds['position_z'])
            ])
        
        logger.info("Starting local optimization (COBYLA)...")
        logger.info(f"Initial guess: {initial_guess}")
        
        # Define bounds as constraints for COBYLA
        constraints = []
        for i, (name, (lb, ub)) in enumerate(self.bounds.items()):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, lb=lb: x[i] - lb  # x >= lb
            })
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, ub=ub: ub - x[i]  # x <= ub
            })
        
        result = minimize(
            self.objective_function,
            initial_guess,
            method='COBYLA',
            constraints=constraints,
            options={
                'maxiter': 20,  # Quick optimization
                'rhobeg': 10.0,  # Initial step size
                'tol': 1e3       # Tolerance in kV/m
            }
        )
        
        return result
    
    def optimize_global(self):
        """
        Global optimization using Differential Evolution.
        Slower but more thorough - finds global minimum.
        
        Returns:
            Optimization result object
        """
        logger.info("Starting global optimization (Differential Evolution)...")
        
        bounds = [
            self.bounds['ring_diameter'],
            self.bounds['tube_diameter'],
            self.bounds['position_z']
        ]
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            strategy='best1bin',
            maxiter=10,           # Population iterations (conservative)
            popsize=5,            # Population size (5*3 = 15 evaluations/iter)
            tol=1e-3,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,         # Don't run local refinement (save time)
            workers=1,            # Serial execution (FEM not thread-safe)
            updating='deferred'
        )
        
        return result
    
    def plot_convergence(self):
        """Plot optimization convergence history."""
        if not self.iteration_history:
            logger.warning("No iteration history to plot")
            return
        
        logger.info("Generating convergence plots...")
        
        iterations = [h['iteration'] for h in self.iteration_history]
        E_max_vals = [h['E_max']/1e3 for h in self.iteration_history]  # kV/m
        
        # Running minimum
        E_min_running = [min(E_max_vals[:i+1]) for i in range(len(E_max_vals))]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Convergence curve
        ax = axes[0, 0]
        ax.plot(iterations, E_max_vals, 'o-', label='E_max', alpha=0.6)
        ax.plot(iterations, E_min_running, 'r-', linewidth=2, 
               label='Best so far')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max E-field (kV/m)')
        ax.set_title('Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Parameter evolution - Ring diameter
        ax = axes[0, 1]
        ring_diameters = [h['ring_diameter'] for h in self.iteration_history]
        ax.plot(iterations, ring_diameters, 'o-', color='blue')
        ax.axhline(self.best_result['params'][0], color='red', 
                  linestyle='--', label='Optimal')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Ring Diameter (mm)')
        ax.set_title('Ring Diameter Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Parameter evolution - Tube diameter
        ax = axes[1, 0]
        tube_diameters = [h['tube_diameter'] for h in self.iteration_history]
        ax.plot(iterations, tube_diameters, 'o-', color='green')
        ax.axhline(self.best_result['params'][1], color='red', 
                  linestyle='--', label='Optimal')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Tube Diameter (mm)')
        ax.set_title('Tube Diameter Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Parameter evolution - Position
        ax = axes[1, 1]
        positions = [h['position_z'] for h in self.iteration_history]
        ax.plot(iterations, positions, 'o-', color='purple')
        ax.axhline(self.best_result['params'][2], color='red', 
                  linestyle='--', label='Optimal')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Position from HV (mm)')
        ax.set_title('Position Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / "optimization_convergence.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"  Saved: {output_path}")
        
        plt.close()
    
    def save_results(self, opt_result):
        """
        Save optimization results to file.
        
        Args:
            opt_result: SciPy optimization result object
        """
        logger.info("Saving optimization results...")
        
        output_path = self.output_dir / "optimization_results.txt"
        
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("   CORONA RING OPTIMIZATION RESULTS - 400 kV INSULATOR\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("OPTIMAL PARAMETERS:\n")
            f.write(f"  Ring Diameter:      {self.best_result['params'][0]:.2f} mm\n")
            f.write(f"  Tube Diameter:      {self.best_result['params'][1]:.2f} mm\n")
            f.write(f"  Position from HV:   {self.best_result['params'][2]:.2f} mm\n")
            f.write(f"\n")
            f.write(f"PERFORMANCE:\n")
            f.write(f"  Max E-field:        {self.best_result['E_max']/1e3:.2f} kV/m\n")
            f.write(f"  IEEE Dry Limit:     4000 kV/m\n")
            f.write(f"  Safety Margin:      {(4e6 - self.best_result['E_max'])/4e6*100:.1f}%\n")
            f.write(f"\n")
            f.write(f"OPTIMIZATION STATISTICS:\n")
            f.write(f"  Total Evaluations:  {self.eval_count}\n")
            f.write(f"  Success:            {opt_result.success}\n")
            f.write(f"  Message:            {opt_result.message}\n")
            f.write("\n")
            
            # Iteration log
            f.write("ITERATION HISTORY:\n")
            f.write(f"{'Iter':<6} {'D (mm)':<10} {'d (mm)':<10} {'z (mm)':<10} {'E_max (kV/m)':<15}\n")
            f.write("-" * 60 + "\n")
            for h in self.iteration_history:
                f.write(f"{h['iteration']:<6} "
                       f"{h['ring_diameter']:<10.2f} "
                       f"{h['tube_diameter']:<10.2f} "
                       f"{h['position_z']:<10.2f} "
                       f"{h['E_max']/1e3:<15.2f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        logger.info(f"  Saved: {output_path}")
        
        # Also save CSV for easy analysis
        import pandas as pd
        df = pd.DataFrame(self.iteration_history)
        df.to_csv(self.output_dir / "optimization_history.csv", index=False)
    
    def run(self, method='local', initial_guess=None):
        """
        Execute optimization.
        
        Args:
            method: 'local' (fast, ~20 min) or 'global' (thorough, ~2 hours)
            initial_guess: Starting point for local optimization
            
        Returns:
            Optimization result
        """
        logger.info("=" * 70)
        logger.info("  CORONA RING OPTIMIZATION FOR 400 kV INSULATOR")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        if method == 'local':
            result = self.optimize_local(initial_guess)
        elif method == 'global':
            result = self.optimize_global()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        elapsed = time.time() - start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("  OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Total evaluations: {self.eval_count}")
        logger.info(f"\nOptimal design:")
        logger.info(f"  Ring Diameter: {self.best_result['params'][0]:.2f} mm")
        logger.info(f"  Tube Diameter: {self.best_result['params'][1]:.2f} mm")
        logger.info(f"  Position: {self.best_result['params'][2]:.2f} mm")
        logger.info(f"  E_max: {self.best_result['E_max']/1e3:.2f} kV/m")
        logger.info("=" * 70)
        
        # Generate plots
        self.plot_convergence()
        
        # Save results
        self.save_results(result)
        
        return result


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize corona ring parameters for 400 kV insulator"
    )
    parser.add_argument(
        "--method",
        choices=['local', 'global'],
        default='local',
        help="Optimization method (local=fast, global=thorough)"
    )
    parser.add_argument(
        "--output",
        default="optimization",
        help="Output directory for results"
    )
    parser.add_argument(
        "--initial",
        nargs=3,
        type=float,
        metavar=('D', 'd', 'z'),
        help="Initial guess: ring_diameter tube_diameter position_z (mm)"
    )
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = CoronaRingOptimizer(output_dir=args.output)
    
    # Run
    try:
        result = optimizer.run(
            method=args.method,
            initial_guess=args.initial
        )
        
        logger.info(f"\n✓ Success! Results saved to: {args.output}/")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠ Optimization interrupted by user")
        optimizer.plot_convergence()
        optimizer.save_results(None)
        return 1
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())