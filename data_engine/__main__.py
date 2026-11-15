"""`python -m data_engine` downloads every required NASA dataset."""
from .lola_downloader import ensure_global_lola, ensure_lola
from .spice_loader import load_kernels

if __name__ == "__main__":
    load_kernels()
    print("SPICE kernels ready")
    print("LOLA DEM ready:", ensure_lola())
    print("LOLA whole-Moon grid ready:", ensure_global_lola())
