"""
COSMOSWeb Galaxy Generator
Load and extract galaxy data from the COSMOSWeb master catalog
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from galgenai.config import load_config
from pathlib import Path


class COSMOSWebCatalog:
    """Class to handle COSMOSWeb galaxy catalog data"""

    def __init__(
            self, 
            catalog_path=None, 
            required_columns=None, 
            required_columns_only=False, 
            warn_flag_cut=0, 
            galaxies_only=True, 
            filter_invalid_mags=False, 
            mag_sentinel=999.0,
        ):
        """
        Initialize the catalog

        Parameters:
        -----------
        catalog_path : str, optional
            Path to the COSMOSWeb FITS catalog file.
            If not provided, it loads Config file (ai_univ_config.yaml)
            The catalog is not included in the package and must be downloaded separately.
        filter_invalid_mags : bool, optional
            If True, filter out galaxies with sentinel magnitude values (default: True)
        mag_sentinel : float, optional
            Sentinel value indicating missing magnitude data (default: 999.0)
        """
        if catalog_path is None:
            # Try config file first
            config_dict = load_config()
            catalog_path = config_dict["cosmos"]["catalog_path"]
            if not Path(catalog_path).exists():
                raise ValueError(
                "The COSMOSWeb catalog is not included in the package.\n"
                "Please download it and specify its path."
            )

        print(f"Loading COSMOSWeb catalog from {catalog_path}...")
        self.catalog_path = catalog_path
        self.data = None
        self.warn_flag_cut = warn_flag_cut
        self.galaxies_only = galaxies_only # If True: No stars/qso objects (comes from LePHARE)
        self.filter_invalid_mags = filter_invalid_mags
        self.mag_sentinel = mag_sentinel
        if required_columns_only:
            if required_columns is None:
                required_columns = [
                    "id", "ra", "dec", "snr_hst-f814w",
                    "mag_model_hsc-g", "mag_model_hsc-r", "mag_model_hsc-i",
                    "mag_model_hsc-z", "mag_model_hsc-y",
                    "radius_sersic", "sersic", "axratio_sersic", "angle_sersic", "zfinal", "type", "warn_flag"
                ]

        self.load_catalog(required_columns=required_columns, required_columns_only=required_columns_only)

    def load_catalog(self, required_columns=None, required_columns_only=False):
        with fits.open(self.catalog_path) as hdul:
            fitsdata_photoetry = hdul[1].data
            fitsdata_redshift = hdul[2].data
            if required_columns_only:
                photometry_cols = [col for col in required_columns if col in fitsdata_photoetry.names]
                redshift_cols = [col for col in required_columns if col in fitsdata_redshift.names]
                photometry = Table({col: fitsdata_photoetry[col] for col in photometry_cols})
                redshift = Table({col: fitsdata_redshift[col] for col in redshift_cols})
            else:
                photometry = Table(fitsdata_photoetry)
                redshift = Table(fitsdata_redshift)
    
        self.data = hstack([photometry, redshift])
        initial_count = len(self.data)

        if self.warn_flag_cut is not None:
            self.data = self.data[self.data["warn_flag"]==self.warn_flag_cut]
            print(f"Applied warn_flag={self.warn_flag_cut} filter: {len(self.data)}/{initial_count} remaining")

        if self.galaxies_only:
            before_count = len(self.data)
            self.data = self.data[self.data["type"]==0] # LePHARE classification 0: galaxy, 1: star, 2: qso
            print(f"Applied galaxies_only filter: {len(self.data)}/{before_count} remaining")

        # Filter out galaxies with invalid magnitudes
        if self.filter_invalid_mags:
            before_count = len(self.data)
            # Find all magnitude columns (mag_model_hsc-*)
            mag_cols = [col for col in self.data.colnames if col.startswith('mag_model_hsc-')]
            if mag_cols:
                mask = np.ones(len(self.data), dtype=bool)
                for mag_col in mag_cols:
                    # Filter out rows where magnitude equals sentinel value
                    col_mask = self.data[mag_col] != self.mag_sentinel
                    mask &= col_mask
                self.data = self.data[mask]
                n_removed = before_count - len(self.data)
                print(f"Filtered {n_removed} galaxies with invalid magnitudes (sentinel={self.mag_sentinel})")
                print(f"Final catalog size: {len(self.data)} galaxies")


    def get_column_names(self):
        """Return all available column names"""
        return self.data.colnames

    def get_galaxies(self, n_galaxies=None, filters=None):
        """
        Extract galaxy data from the photometry catalog

        Parameters:
        -----------
        n_galaxies : int, optional
            Number of galaxies to return (None = all)
        filters : dict, optional
            Dictionary of filters to apply, e.g., {'mag_auto': (20, 25)}

        Returns:
        --------
        astropy.table.Table
            Table containing galaxy data
        """
        galaxies = self.data.copy()

        # Apply filters if provided
        if filters:
            for col, (min_val, max_val) in filters.items():
                if col in galaxies.colnames:
                    mask = (galaxies[col] >= min_val) & (galaxies[col] <= max_val)
                    galaxies = galaxies[mask]
                    print(f"Applied filter {col}: [{min_val}, {max_val}] -> {len(galaxies)} objects remaining")

        # Limit number if specified
        if n_galaxies and n_galaxies < len(galaxies):
            galaxies = galaxies[:n_galaxies]

        return galaxies

    def find_matching_col_names(self, pattern, verbose=0):
        matching_cols = [col for col in self.get_column_names() if pattern in col.lower()]

        if verbose:
            print(f"Found {len(matching_cols)} column(s) containing '{pattern}':")
            for col in matching_cols:
                print(f"  {col}")
        
        return matching_cols





def main():
    """Example usage"""

    # Initialize catalog
    catalog = COSMOSWebCatalog()

    # Show summary
    catalog.get_summary_statistics()

    # Example 1: Get first 100 galaxies
    print("\n=== Example 1: First 100 galaxies ===")
    galaxies_100 = catalog.get_galaxies(n_galaxies=100)
    print(f"Retrieved {len(galaxies_100)} galaxies")

    # Example 2: Filter by redshift
    print("\n=== Example 2: Filter by redshift ===")
    high_z_galaxies = catalog.filter_by_redshift(z_min=2.0, z_max=4.0)

    # Example 3: Get specific properties
    print("\n=== Example 3: Extract galaxy properties ===")
    props = catalog.get_galaxy_properties()
    print(f"Properties Table shape: ({len(props)}, {len(props.colnames)})")
    print(f"Columns: {props.colnames}")
    print("\nFirst 5 rows:")
    print(props[:5])

    return catalog
