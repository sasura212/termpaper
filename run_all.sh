#!/bin/bash
################################################################################
# Automated Pipeline for 400 kV Insulator Electrostatic Simulation
# 
# Reference: Corona Ring Improvement to Surface Electric Field Stress 
#            Mitigation of 400 kV Composite Insulator (2024)
################################################################################

set -e  # Exit on error

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Parse command-line arguments
SKIP_MESH=false
SKIP_SOLVE=false
SKIP_POST=false
SKIP_OPT=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-mesh)
            SKIP_MESH=true
            shift
            ;;
        --skip-solve)
            SKIP_SOLVE=true
            shift
            ;;
        --skip-post)
            SKIP_POST=true
            shift
            ;;
        --skip-opt)
            SKIP_OPT=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./run_all.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-mesh    Skip mesh generation (use existing mesh)"
            echo "  --skip-solve   Skip FEM solving"
            echo "  --skip-post    Skip post-processing"
            echo "  --skip-opt     Skip optimization"
            echo "  --quick        Quick mode (simplified mesh, fast solve)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Header
################################################################################

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  400 kV COMPOSITE INSULATOR ELECTROSTATIC SIMULATION PIPELINE        ║"
echo "║                                                                       ║"
echo "║  Reference: Corona Ring Improvement to Surface Electric Field        ║"
echo "║             Stress Mitigation of 400 kV Composite Insulator          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

START_TIME=$(date +%s)

################################################################################
# Step 0: Environment Check
################################################################################

print_step "Checking environment..."

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION detected"

# Check critical dependencies
MISSING_DEPS=()

if ! python -c "import gmsh" &> /dev/null; then
    MISSING_DEPS+=("gmsh")
fi

if ! python -c "import meshio" &> /dev/null; then
    MISSING_DEPS+=("meshio")
fi

if ! python -c "import numpy" &> /dev/null; then
    MISSING_DEPS+=("numpy")
fi

if ! python -c "import scipy" &> /dev/null; then
    MISSING_DEPS+=("scipy")
fi

if ! python -c "import matplotlib" &> /dev/null; then
    MISSING_DEPS+=("matplotlib")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    print_warning "Missing dependencies: ${MISSING_DEPS[*]}"
    print_warning "Installing with pip..."
    pip install "${MISSING_DEPS[@]}"
fi

# Check FEniCS (optional but recommended)
if ! python -c "import dolfin" &> /dev/null; then
    print_warning "FEniCS not installed - will use analytical approximation"
    print_warning "For accurate results, install FEniCS:"
    print_warning "  conda install -c conda-forge fenics"
else
    print_success "FEniCS detected - full FEM solver available"
fi

# Create directories
mkdir -p geometry results figures optimization logs

################################################################################
# Step 1: Mesh Generation
################################################################################

if [ "$SKIP_MESH" = false ]; then
    print_step "STEP 1: Generating 3D mesh..."
    
    # Check for input STL
    if [ -f "geometry/insulator_model.stl" ]; then
        print_success "Found input STL: geometry/insulator_model.stl"
        MESH_ARGS=""
    else
        print_warning "No STL found - generating default geometry"
        MESH_ARGS="--generate-default"
    fi
    
    # Set mesh size based on mode
    if [ "$QUICK_MODE" = true ]; then
        MESH_SIZE="50.0"  # Coarse mesh (fast)
        print_warning "Quick mode: Using coarse mesh (size=50mm)"
    else
        MESH_SIZE="20.0"  # Fine mesh (accurate)
    fi
    
    if python mesh_utils.py $MESH_ARGS --mesh-size $MESH_SIZE 2>&1 | tee logs/mesh_generation.log; then
        print_success "Mesh generation complete"
    else
        print_error "Mesh generation failed - check logs/mesh_generation.log"
        exit 1
    fi
else
    print_step "STEP 1: Skipping mesh generation (using existing mesh)"
fi

echo ""

################################################################################
# Step 2: Solve Electrostatics
################################################################################

if [ "$SKIP_SOLVE" = false ]; then
    print_step "STEP 2: Solving electrostatic field..."
    
    if python solve_electrostatics.py --voltage 400 2>&1 | tee logs/solver.log; then
        print_success "FEM solution complete"
    else
        print_error "Solver failed - check logs/solver.log"
        exit 1
    fi
else
    print_step "STEP 2: Skipping FEM solve (using existing results)"
fi

echo ""

################################################################################
# Step 3: Post-Processing
################################################################################

if [ "$SKIP_POST" = false ]; then
    print_step "STEP 3: Post-processing and visualization..."
    
    if python postprocess.py 2>&1 | tee logs/postprocess.log; then
        print_success "Post-processing complete"
        
        # Display key results
        if [ -f "results/solution_summary.txt" ]; then
            echo ""
            cat results/solution_summary.txt
        fi
    else
        print_error "Post-processing failed - check logs/postprocess.log"
        exit 1
    fi
else
    print_step "STEP 3: Skipping post-processing"
fi

echo ""

################################################################################
# Step 4: Corona Ring Optimization (Optional)
################################################################################

if [ "$SKIP_OPT" = false ]; then
    print_step "STEP 4: Optimizing corona ring parameters..."
    
    # Ask user for confirmation (optimization takes time)
    echo ""
    echo "Optimization will take approximately:"
    if [ "$QUICK_MODE" = true ]; then
        echo "  - Local method: 15-20 minutes"
        echo "  - Global method: 1-2 hours"
    else
        echo "  - Local method: 30-45 minutes"
        echo "  - Global method: 2-4 hours"
    fi
    echo ""
    read -p "Continue with optimization? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Choose method
        read -p "Select method - (l)ocal or (g)lobal? [L/g] " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Gg]$ ]]; then
            OPT_METHOD="global"
        else
            OPT_METHOD="local"
        fi
        
        if python optimize_corona_ring.py --method $OPT_METHOD 2>&1 | tee logs/optimization.log; then
            print_success "Optimization complete"
            
            # Display optimal parameters
            if [ -f "optimization/optimization_results.txt" ]; then
                echo ""
                head -20 optimization/optimization_results.txt
            fi
        else
            print_error "Optimization failed - check logs/optimization.log"
        fi
    else
        print_warning "Skipping optimization"
    fi
else
    print_step "STEP 4: Skipping optimization"
fi

echo ""

################################################################################
# Summary
################################################################################

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                      PIPELINE COMPLETE                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
print_success "Total execution time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Generated outputs:"
echo "  📁 geometry/        - Mesh files (.msh, .xdmf)"
echo "  📁 results/         - Solution files (voltage, E-field)"
echo "  📁 figures/         - Plots and visualizations"
echo "  📁 optimization/    - Optimization results"
echo "  📁 logs/            - Execution logs"
echo ""
echo "Next steps:"
echo "  1. Review figures/ for electric field distribution"
echo "  2. Check results/solution_summary.txt for compliance"
echo "  3. If E-field exceeds limits, run optimization"
echo "  4. Use ParaView to visualize .xdmf files in 3D"
echo ""
print_success "All done! 🎉"
echo ""