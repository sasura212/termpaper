"""
Installation Validation Script

Quick test to verify all dependencies are correctly installed
and the simulation pipeline is ready to run.

Usage: python test_installation.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    display_name = package_name or module_name
    try:
        __import__(module_name)
        logger.info(f"  ✓ {display_name}")
        return True
    except ImportError:
        logger.error(f"  ✗ {display_name} - NOT INSTALLED")
        return False

def main():
    """Run all installation tests."""
    
    print("\n" + "="*70)
    print("  400 kV INSULATOR SIMULATION - INSTALLATION TEST")
    print("="*70 + "\n")
    
    all_ok = True
    
    # Core dependencies
    print("📦 Core Dependencies:")
    all_ok &= test_import("numpy", "NumPy")
    all_ok &= test_import("scipy", "SciPy")
    all_ok &= test_import("matplotlib", "Matplotlib")
    
    # Mesh generation
    print("\n🔧 Mesh Generation:")
    all_ok &= test_import("gmsh", "Gmsh")
    all_ok &= test_import("meshio", "MeshIO")
    
    # FEM solver (optional but critical)
    print("\n⚙️  FEM Solver:")
    fenics_ok = test_import("dolfin", "FEniCS (dolfin)")
    if not fenics_ok:
        logger.warning("  ⚠️  FEniCS not installed - will use analytical fallback")
        logger.warning("  ⚠️  Install for accurate results: conda install -c conda-forge fenics")
    
    # Visualization
    print("\n📊 Visualization:")
    all_ok &= test_import("pyvista", "PyVista")
    test_import("vtk", "VTK")  # Optional, PyVista dependency
    
    # Utilities
    print("\n🛠️  Utilities:")
    all_ok &= test_import("tqdm", "tqdm")
    test_import("pandas", "Pandas")  # Optional
    
    # Test file I/O
    print("\n📁 File System:")
    from pathlib import Path
    dirs = ['geometry', 'results', 'figures', 'optimization', 'logs']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        logger.info(f"  ✓ Directory: {d}/")
    
    # Summary
    print("\n" + "="*70)
    if all_ok and fenics_ok:
        print("  ✅ ALL TESTS PASSED - READY TO RUN")
        print("\n  Next steps:")
        print("    1. Place your STL in geometry/insulator_model.stl")
        print("    2. Run: ./run_all.sh")
    elif all_ok:
        print("  ⚠️  CORE DEPENDENCIES OK - FENICS MISSING")
        print("\n  Current status:")
        print("    • Can generate meshes ✓")
        print("    • Will use analytical solver (less accurate)")
        print("\n  Recommended:")
        print("    • Install FEniCS: conda install -c conda-forge fenics")
    else:
        print("  ❌ INSTALLATION INCOMPLETE")
        print("\n  Missing dependencies detected!")
        print("  Install with: pip install -r requirements.txt")
        return 1
    
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())