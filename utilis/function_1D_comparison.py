import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings


def safe_trapezoid(y, x):
    """
    Safe wrapper for trapezoidal integration that handles NumPy version compatibility.
    Uses np.trapezoid if available (NumPy >= 2.0), otherwise falls back to np.trapz.
    """
    try:
        # Try the new function first (NumPy >= 2.0)
        return np.trapezoid(y, x)
    except AttributeError:
        # Fallback to the old function (NumPy < 2.0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return np.trapz(y, x)

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings


def safe_trapezoid(y, x):
    """
    Safe wrapper for trapezoidal integration that handles NumPy version compatibility.
    Uses np.trapezoid if available (NumPy >= 2.0), otherwise falls back to np.trapz.
    """
    try:
        # Try the new function first (NumPy >= 2.0)
        return np.trapezoid(y, x)
    except AttributeError:
        # Fallback to the old function (NumPy < 2.0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return np.trapz(y, x)


class ReferenceBasedPeakComparison:
    """
    System for comparing peak shapes using averaged reference profiles
    fitted with Gaussian functions using GMM approach.
    """
    
    def __init__(self, high_res_factor=10):
        """
        Parameters:
        -----------
        high_res_factor : int
            Factor to increase resolution for reference fitting
        """
        self.high_res_factor = high_res_factor
        self.reference_profiles = {}
        self.fitted_references = {}
        self.comparison_results = {}
    
    def gaussian(self, x, amp, center, sigma):
        """Single Gaussian function."""
        return amp * np.exp(-(x - center)**2 / (2 * sigma**2))
    
    def multi_gaussian(self, x, *params):
        """Multiple Gaussian functions."""
        result = np.zeros_like(x)
        n_peaks = len(params) // 3
        for i in range(n_peaks):
            amp, center, sigma = params[i*3:(i+1)*3]
            result += self.gaussian(x, amp, center, sigma)
        return result
    
    def create_averaged_reference(self, data, x_axis, axis_name):
        """
        Create averaged 1D profile from all data points.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data with shape (NoPoints, NoAxis1, NoAxis2, NoAxis3)
        x_axis : numpy.ndarray
            Coordinate array for the axis of interest
        axis_name : str
            Name identifier for this axis ('axis1', 'axis2', or 'axis3')
        
        Returns:
        --------
        dict : Reference profile information
        """
        NoPoints = data.shape[0]
        
        # Extract 1D profiles for all points
        if axis_name == 'axis1':
            profiles = np.sum(data, axis=(2, 3))  # Sum over axis2 and axis3
        elif axis_name == 'axis2':
            profiles = np.sum(data, axis=(1, 3))  # Sum over axis1 and axis3
        elif axis_name == 'axis3':
            profiles = np.sum(data, axis=(1, 2))  # Sum over axis1 and axis2
        else:
            raise ValueError("axis_name must be 'axis1', 'axis2', or 'axis3'")
        
        # Calculate averaged profile
        avg_profile = np.mean(profiles, axis=0)
        
        # Store reference
        self.reference_profiles[axis_name] = {
            'x_axis': x_axis,
            'average_profile': avg_profile,
            'individual_profiles': profiles,
            'std_profile': np.std(profiles, axis=0)
        }
        
        return self.reference_profiles[axis_name]

    def fit_reference_gaussian(self, axis_name, max_peaks=3, prominence_threshold=0.1, 
                            min_peak_distance=None, height_threshold=0.05, 
                            manual_peak_positions=None, width_method='adaptive', width_factor=1.0):
        """
        Enhanced version of your existing fit_reference_gaussian with width control.
        
        Additional Parameters:
        ---------------------
        width_method : str
            Width estimation method: 'half_max', 'quarter_max', 'spacing', 'curvature', 'adaptive'
        width_factor : float
            Multiplicative factor to adjust estimated widths (1.0 = no adjustment)
        """
        if axis_name not in self.reference_profiles:
            raise ValueError(f"No reference profile found for {axis_name}")
        
        ref_data = self.reference_profiles[axis_name]
        x_axis = ref_data['x_axis']
        avg_profile = ref_data['average_profile']
        
        if np.max(avg_profile) <= 0:
            return {'success': False, 'reason': 'No signal in averaged profile'}
        
        # Create high-resolution x-axis for fitting
        x_min, x_max = np.min(x_axis), np.max(x_axis)
        x_high_res = np.linspace(x_min, x_max, len(x_axis) * self.high_res_factor)
        
        # Interpolate average profile to high resolution
        try:
            f_interp = interp1d(x_axis, avg_profile, kind='cubic', bounds_error=False, fill_value=0)
            avg_profile_high_res = f_interp(x_high_res)
            avg_profile_high_res = np.maximum(avg_profile_high_res, 0)  # Ensure non-negative
        except:
            # Fallback to linear interpolation
            f_interp = interp1d(x_axis, avg_profile, kind='linear', bounds_error=False, fill_value=0)
            avg_profile_high_res = f_interp(x_high_res)
            avg_profile_high_res = np.maximum(avg_profile_high_res, 0)
        
        # Handle manual peak positions first
        if manual_peak_positions is not None:
            print(f"  - Using manual peak positions: {manual_peak_positions}")
            # Convert manual positions to indices in high-res array
            peaks = []
            for pos in manual_peak_positions:
                idx = np.argmin(np.abs(x_high_res - pos))
                peaks.append(idx)
            peaks = np.array(peaks)
            n_peaks = len(peaks)
            print(f"  - Manual peaks converted to indices: {peaks}")
        else:
            # Automatic peak detection
            y_max = np.max(avg_profile_high_res)
            y_norm = avg_profile_high_res / y_max
            
            # Set minimum distance between peaks if not provided
            if min_peak_distance is None:
                min_peak_distance = len(x_high_res) // (max_peaks * 4)  # Adaptive distance
            
            # Detect peaks with multiple criteria
            peaks, properties = find_peaks(
                y_norm, 
                prominence=prominence_threshold,
                height=height_threshold,
                distance=min_peak_distance
            )
            
            print(f"  - Detected {len(peaks)} peaks in {axis_name}")
            
            # Determine number of peaks to fit
            if len(peaks) == 0:
                print(f"  - No peaks detected, using global maximum")
                peaks = [np.argmax(avg_profile_high_res)]
                n_peaks = 1
            else:
                # Sort peaks by prominence and take the most significant ones
                if len(peaks) > max_peaks:
                    peak_prominences = properties['prominences']
                    sorted_indices = np.argsort(peak_prominences)[::-1]  # Sort by prominence, descending
                    peaks = peaks[sorted_indices[:max_peaks]]
                    peaks = np.sort(peaks)  # Sort by position for consistency
                n_peaks = len(peaks)
        
        print(f"  - Fitting {n_peaks} peaks at positions: {[x_high_res[p] for p in peaks]}")
        print(f"  - Using width method: {width_method}, width factor: {width_factor}")
        
        # Use GMM fitting approach
        if n_peaks == 1:
            fit_result = self._fit_single_gaussian(x_high_res, avg_profile_high_res, peaks)
        else:
            # MODIFIED: Pass width control parameters to GMM fitting
            fit_result = self._fit_gaussians_gmm(
                x_high_res, avg_profile_high_res, n_peaks, peaks, 
                width_method=width_method, width_factor=width_factor
            )
        
        if fit_result['success']:
            # Store fitted reference
            self.fitted_references[axis_name] = {
                'x_high_res': x_high_res,
                'fitted_curve': fit_result['fitted_curve'],
                'parameters': fit_result['parameters'],
                'n_peaks': n_peaks,
                'r_squared': fit_result['r_squared'],
                'original_x': x_axis,
                'original_profile': avg_profile,
                'detected_peaks': peaks,  # Store peak positions
                'width_method': width_method,  # Store width method used
                'width_factor': width_factor   # Store width factor used
            }
        
        return fit_result

    def _fit_single_gaussian(self, x_values, y_values, detected_peaks=None):
        """Fit a single Gaussian using scipy.optimize.curve_fit."""
        try:
            # Use global maximum if no peaks provided
            if detected_peaks is not None and len(detected_peaks) > 0:
                center_guess = x_values[detected_peaks[0]]
                amp_guess = y_values[detected_peaks[0]]
            else:
                max_idx = np.argmax(y_values)
                center_guess = x_values[max_idx]
                amp_guess = y_values[max_idx]
            
            sigma_guess = (x_values[-1] - x_values[0]) / 6
            
            initial_guess = [amp_guess, center_guess, sigma_guess]
            bounds = ([0, x_values[0], 0], [np.inf, x_values[-1], x_values[-1] - x_values[0]])
            
            popt, pcov = curve_fit(
                self.gaussian, x_values, y_values, 
                p0=initial_guess, bounds=bounds, maxfev=3000
            )
            
            fitted_curve = self.gaussian(x_values, *popt)
            parameters = [{'amplitude': popt[0], 'center': popt[1], 'sigma': abs(popt[2])}]
            
            # Calculate R-squared
            ss_res = np.sum((y_values - fitted_curve) ** 2)
            ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'success': True,
                'fitted_curve': fitted_curve,
                'parameters': parameters,
                'r_squared': r_squared,
                'method': 'single_gaussian'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


    # Alternative: Even simpler fix - just modify the existing fit function slightly
    def _fit_gaussians_gmm(self, x_values, y_values, n_peaks, detected_peaks, 
                                            width_method='adaptive', width_factor=1.0):
        """
        Modified version of your existing _fit_gaussians_gmm with width control.
        """
        # Clean input data
        y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        try:
            print(f"    - GMM fitting with {n_peaks} components and width control")
            
            # Step 1: Convert 1D profile to 2D weighted samples for GMM
            samples, weights, norm_factor = self._profile_to_samples(x_values, y_values)
            
            if len(samples) < n_peaks * 3:  # Need enough samples
                print(f"    - Insufficient samples ({len(samples)}) for GMM with {n_peaks} components")
                return {'success': False, 'error': 'Insufficient samples for GMM'}
            
            print(f"    - Created {len(samples)} weighted samples, normalization factor: {norm_factor:.2e}")
            
            # Step 2: Initialize GMM with width-controlled starting points
            gmm = GaussianMixture(
                n_components=n_peaks,
                covariance_type='full',  # Allow different widths
                init_params='k-means++',  # Smart initialization
                max_iter=200,
                n_init=5,  # Try 5 different initializations
                random_state=42
            )
            
            # MODIFIED: Use width-controlled initialization
            if detected_peaks is not None and len(detected_peaks) >= n_peaks:
                peak_positions = [x_values[detected_peaks[i]] for i in range(n_peaks)]
                gmm = self._initialize_gmm(
                    gmm, samples, peak_positions, n_peaks,
                    x_values=x_values, y_values=y_values, detected_peaks=detected_peaks,
                    width_method=width_method, width_factor=width_factor
                )
                print(f"    - GMM initialized with width-controlled peak positions: {peak_positions}")
            
            # Step 3: Fit GMM (handle version compatibility) - SAME AS ORIGINAL
            try:
                # Try with sample_weight (newer scikit-learn versions)
                gmm.fit(samples.reshape(-1, 1), sample_weight=weights)
            except TypeError:
                # Fallback for older scikit-learn versions without sample_weight support
                print("    - sample_weight not supported, using unweighted GMM fitting")
                # Create weighted samples by replicating data points based on weights
                weighted_samples = self._create_weighted_samples(samples, weights)
                gmm.fit(weighted_samples.reshape(-1, 1))
            
            # Step 4: Extract Gaussian parameters - SAME AS ORIGINAL
            parameters = []
            individual_gaussians = []
            
            for i in range(n_peaks):
                # GMM provides: mean, covariance, weight
                center = gmm.means_[i, 0]
                variance = gmm.covariances_[i, 0, 0]
                sigma = np.sqrt(variance)
                weight = gmm.weights_[i]
                
                # Convert weight back to amplitude using normalization factor
                total_area = norm_factor
                amplitude = weight * total_area / (sigma * np.sqrt(2 * np.pi))
                
                parameters.append({
                    'amplitude': amplitude,
                    'center': center,
                    'sigma': sigma,
                    'weight': weight
                })
                
                # Create individual Gaussian function
                gaussian_func = lambda x, a=amplitude, c=center, s=sigma: a * np.exp(-(x - c)**2 / (2 * s**2))
                individual_gaussians.append(gaussian_func)
                
                print(f"    - Component {i+1}: center={center:.3f}, sigma={sigma:.4f}, amplitude={amplitude:.2f}, weight={weight:.3f}")
            
            # Sort parameters by center position
            sorted_indices = np.argsort([p['center'] for p in parameters])
            parameters = [parameters[i] for i in sorted_indices]
            individual_gaussians = [individual_gaussians[i] for i in sorted_indices]
            
            # Step 5: Create combined fitted curve - SAME AS ORIGINAL
            fitted_curve = np.zeros_like(x_values)
            for gaussian_func in individual_gaussians:
                fitted_curve += gaussian_func(x_values)
            
            # Step 6: Calculate R-squared - SAME AS ORIGINAL
            ss_res = np.sum((y_values - fitted_curve) ** 2)
            ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Step 7: Quality assessment - SAME AS ORIGINAL
            try:
                aic = gmm.aic(samples.reshape(-1, 1))
                bic = gmm.bic(samples.reshape(-1, 1))
                log_likelihood = gmm.score(samples.reshape(-1, 1))
            except:
                aic = np.nan
                bic = np.nan  
                log_likelihood = np.nan
            
            print(f"    - Width-controlled GMM fitting completed: R² = {r_squared:.4f}")
            print(f"    - Model quality: AIC = {aic:.2f}, BIC = {bic:.2f}, LogLik = {log_likelihood:.2f}")
            
            return {
                'success': True,
                'fitted_curve': fitted_curve,
                'parameters': parameters,
                'r_squared': r_squared,
                'individual_gaussians': individual_gaussians,
                'gmm_model': gmm,
                'model_quality': {
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': log_likelihood
                },
                'method': f'GMM_with_{width_method}_width'
            }
            
        except Exception as e:
            print(f"    - Width-controlled GMM fitting failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _create_weighted_samples(self, samples, weights, max_samples=10000):
        """
        Create weighted samples by replicating data points for older scikit-learn versions.
        
        Parameters:
        -----------
        samples : array
            Original sample positions
        weights : array
            Weights for each sample
        max_samples : int
            Maximum number of samples to generate (to avoid memory issues)
        
        Returns:
        --------
        array : Weighted samples with replications
        """
        # Normalize weights to get replication counts
        total_weight = np.sum(weights)
        normalized_weights = weights / total_weight
        
        # Calculate number of replications for each sample
        replications = np.round(normalized_weights * max_samples).astype(int)
        
        # Ensure at least one replication for non-zero weights
        replications[normalized_weights > 0] = np.maximum(replications[normalized_weights > 0], 1)
        
        # Create weighted samples
        weighted_samples = []
        for i, (sample, rep_count) in enumerate(zip(samples, replications)):
            if rep_count > 0:
                weighted_samples.extend([sample] * rep_count)
        
        return np.array(weighted_samples)
    
    def _profile_to_samples(self, x_values, y_values, min_intensity_fraction=0.01):
        """
        Convert 1D intensity profile to weighted samples for GMM.
        Enhanced to better preserve peak intensity relationships.
        """
        # Remove baseline
        baseline = np.min(y_values)
        y_corrected = y_values - baseline
        
        # Filter out very low intensity points
        max_intensity = np.max(y_corrected)
        threshold = max_intensity * min_intensity_fraction
        
        valid_mask = y_corrected > threshold
        x_samples = x_values[valid_mask]
        y_samples = y_corrected[valid_mask]
        
        # Calculate total area for normalization
        total_area = safe_trapezoid(y_corrected, x_values)
        
        # ENHANCEMENT: Apply power scaling to emphasize intensity differences
        # This helps GMM better preserve relative peak intensities
        power_factor = 0.7  # Values < 1 reduce intensity differences, > 1 enhance them
        y_samples_scaled = np.power(y_samples / max_intensity, power_factor) * max_intensity
        
        # Normalize weights to sum to 1 (probability distribution)
        weights = y_samples_scaled / np.sum(y_samples_scaled)
        
        return x_samples, weights, total_area
    
    def _initialize_gmm(self, gmm, samples, peak_positions, n_peaks, 
                        x_values=None, y_values=None, detected_peaks=None, 
                        width_method='height_based', width_factor=1.0):
        """
        Enhanced GMM initialization with width control.
        
        Parameters:
        -----------
        width_method : str
            Method for width estimation: 'half_max', 'quarter_max', 'spacing', 'curvature', 'adaptive'
        width_factor : float
            Multiplicative factor to adjust all widths (1.0 = no adjustment)
        """
        try:
            # Create initial means
            means_init = np.array(peak_positions).reshape(-1, 1)
            
            # Choose width estimation method
            if width_method in ['height_based', 'position_based', 'intensity_ratio', 'progressive', 'random_variation']:
                # Use diverse width methods
                sigmas = self._estimate_peak_widths_diverse(
                    x_values, y_values, detected_peaks, width_method, width_factor
                )
            elif width_method == 'data_based':
                # Use actual data shape analysis
                sigmas = self._estimate_peak_widths_from_data(
                    x_values, y_values, detected_peaks, diversify=True
                )
            else:
                # Fallback to original method but add diversity
                sigmas = self._estimate_peak_widths_simple(
                    x_values, y_values, detected_peaks, width_method, width_factor
                )
                
                # Add diversity if all widths are too similar
                if n_peaks > 1 and np.std(sigmas) / np.mean(sigmas) < 0.1:  # Less than 10% variation
                    print(f"    - Adding diversity to similar widths")
                    diversity_factors = np.linspace(0.7, 1.3, n_peaks)
                    sigmas = sigmas * diversity_factors
            
            # Create covariance matrices
            covariances_init = np.array([[[sigma**2]] for sigma in sigmas])
            
            # Set initial parameters
            gmm.means_ = means_init
            gmm.covariances_ = covariances_init
            gmm.precisions_cholesky_ = np.array([[[1.0/sigma]] for sigma in sigmas])
            
            print(f"    - GMM initialized with diverse widths: {sigmas}")
            return gmm
            
        except Exception as e:
            print(f"    - Diverse GMM initialization failed, using default: {str(e)}")
            return gmm

    def _estimate_peak_widths_diverse(self, x_values, y_values, detected_peaks, method='height_based', width_factor=1.0):
        """
        Create diverse width estimates to avoid identical initialization.
        
        Parameters:
        -----------
        method : str
            'height_based' - wider peaks for taller peaks
            'position_based' - different widths based on peak position
            'intensity_ratio' - widths proportional to peak intensity
            'progressive' - systematically different widths
            'random_variation' - add controlled randomness
        """
        n_peaks = len(detected_peaks)
        peak_heights = [y_values[idx] for idx in detected_peaks]
        peak_positions = [x_values[idx] for idx in detected_peaks]
        
        # Base width estimate
        if n_peaks > 1:
            avg_spacing = np.mean(np.diff(sorted(peak_positions)))
            base_width = avg_spacing / 4  # Conservative base
        else:
            base_width = (x_values[-1] - x_values[0]) / 10
        
        sigmas = np.zeros(n_peaks)
        
        if method == 'height_based':
            # Taller peaks get wider widths (more prominent = more spread)
            max_height = max(peak_heights)
            min_height = min(peak_heights)
            height_range = max_height - min_height
            
            for i, height in enumerate(peak_heights):
                if height_range > 0:
                    # Scale factor from 0.5 to 1.5 based on height
                    height_factor = 0.5 + (height - min_height) / height_range
                    sigmas[i] = base_width * height_factor * width_factor
                else:
                    sigmas[i] = base_width * width_factor
                    
            print(f"    - Height-based widths: tall peaks = wide, short peaks = narrow")
        
        elif method == 'position_based':
            # Different widths based on position (left=narrow, center=medium, right=wide)
            min_pos = min(peak_positions)
            max_pos = max(peak_positions)
            pos_range = max_pos - min_pos
            
            for i, pos in enumerate(peak_positions):
                if pos_range > 0:
                    # Scale factor from 0.6 to 1.4 based on position
                    pos_factor = 0.6 + 0.8 * (pos - min_pos) / pos_range
                    sigmas[i] = base_width * pos_factor * width_factor
                else:
                    sigmas[i] = base_width * width_factor
                    
            print(f"    - Position-based widths: left=narrow, right=wide")
        
        elif method == 'intensity_ratio':
            # Widths inversely proportional to intensity (short peaks = wide, tall peaks = narrow)
            max_height = max(peak_heights)
            
            for i, height in enumerate(peak_heights):
                # Inverse relationship: lower intensity = wider peak
                intensity_factor = 0.7 + 0.6 * (max_height - height) / max_height
                sigmas[i] = base_width * intensity_factor * width_factor
                
            print(f"    - Intensity-ratio widths: short peaks = wide, tall peaks = narrow")
        
        elif method == 'progressive':
            # Systematically increasing widths
            factors = np.linspace(0.6, 1.4, n_peaks)
            for i in range(n_peaks):
                sigmas[i] = base_width * factors[i] * width_factor
                
            print(f"    - Progressive widths: {factors}")
        
        elif method == 'random_variation':
            # Add controlled random variation
            np.random.seed(42)  # Reproducible
            random_factors = np.random.uniform(0.7, 1.3, n_peaks)
            for i in range(n_peaks):
                sigmas[i] = base_width * random_factors[i] * width_factor
                
            print(f"    - Random variation widths: {random_factors}")
        
        else:
            # Default: equal widths (your current behavior)
            sigmas.fill(base_width * width_factor)
        
        # Ensure minimum width
        min_width = (x_values[1] - x_values[0]) * 2
        sigmas = np.maximum(sigmas, min_width)
        
        print(f"    - Final diverse sigmas: {sigmas}")
        return sigmas

    def _estimate_peak_widths_from_data(self, x_values, y_values, detected_peaks, diversify=True):
        """
        Estimate widths directly from the actual peak shapes in data.
        """
        n_peaks = len(detected_peaks)
        sigmas = np.zeros(n_peaks)
        
        for i, peak_idx in enumerate(detected_peaks):
            try:
                # Get peak height
                peak_height = y_values[peak_idx]
                
                # Define search window around peak
                window_size = min(20, len(y_values) // 8)
                start_idx = max(0, peak_idx - window_size)
                end_idx = min(len(y_values), peak_idx + window_size)
                
                x_local = x_values[start_idx:end_idx]
                y_local = y_values[start_idx:end_idx]
                
                # Find where intensity drops to different fractions
                fractions = [0.8, 0.6, 0.4, 0.2]  # Different thresholds
                width_estimates = []
                
                for frac in fractions:
                    threshold = peak_height * frac
                    
                    # Find left boundary
                    left_idx = peak_idx - start_idx  # Adjust for local array
                    for j in range(left_idx, 0, -1):
                        if y_local[j] <= threshold:
                            left_boundary = x_local[j]
                            break
                    else:
                        left_boundary = x_local[0]
                    
                    # Find right boundary  
                    for j in range(left_idx, len(x_local)):
                        if y_local[j] <= threshold:
                            right_boundary = x_local[j]
                            break
                    else:
                        right_boundary = x_local[-1]
                    
                    # Calculate width at this threshold
                    width = right_boundary - left_boundary
                    if width > 0:
                        # Convert to approximate sigma (different conversion for different fractions)
                        if frac == 0.6:  # Close to half-max
                            sigma_est = width / 2.354  # FWHM conversion
                        else:
                            sigma_est = width / (2 * np.sqrt(2 * np.log(1/frac)))
                        width_estimates.append(sigma_est)
                
                # Use median of estimates
                if width_estimates:
                    sigmas[i] = np.median(width_estimates)
                else:
                    # Fallback
                    sigmas[i] = (x_values[-1] - x_values[0]) / (n_peaks * 8)
                    
            except Exception as e:
                # Fallback for individual peak
                sigmas[i] = (x_values[-1] - x_values[0]) / (n_peaks * 8)
        
        # Add diversity if requested
        if diversify and n_peaks > 1:
            # Add small variations based on peak characteristics
            peak_heights = [y_values[idx] for idx in detected_peaks]
            height_ratios = np.array(peak_heights) / max(peak_heights)
            
            # Modify widths slightly based on height (±20% variation)
            diversity_factors = 0.8 + 0.4 * height_ratios
            sigmas = sigmas * diversity_factors
            
            print(f"    - Added diversity based on peak heights: {diversity_factors}")
        
        print(f"    - Data-based width estimates: {sigmas}")
        return sigmas



    def _estimate_peak_widths_simple(self, x_values, y_values, detected_peaks, method='half_max', width_factor=1.0):
        """
        Simple methods to estimate peak widths for better GMM initialization.
        
        Parameters:
        -----------
        x_values : array
            X-axis coordinates
        y_values : array
            Y-values (intensity profile)
        detected_peaks : array
            Peak indices
        method : str
            Width estimation method: 'half_max', 'quarter_max', 'spacing', 'curvature', 'adaptive'
        width_factor : float
            Multiplicative factor to adjust estimated widths (1.0 = no adjustment)
        
        Returns:
        --------
        array : Estimated sigma values for each peak
        """
        n_peaks = len(detected_peaks)
        sigmas = np.zeros(n_peaks)
        
        print(f"    - Estimating widths using '{method}' method with factor {width_factor}")
        
        for i, peak_idx in enumerate(detected_peaks):
            try:
                if method == 'half_max':
                    sigma = self._width_from_half_max(x_values, y_values, peak_idx)
                elif method == 'quarter_max':
                    sigma = self._width_from_quarter_max(x_values, y_values, peak_idx)
                elif method == 'spacing':
                    sigma = self._width_from_spacing(x_values, detected_peaks, i)
                elif method == 'curvature':
                    sigma = self._width_from_curvature(x_values, y_values, peak_idx)
                elif method == 'adaptive':
                    sigma = self._width_adaptive(x_values, y_values, peak_idx, detected_peaks, i)
                else:
                    # Default fallback
                    sigma = (x_values[-1] - x_values[0]) / (n_peaks * 6)
                
                # Apply width factor and ensure minimum width
                sigma = sigma * width_factor
                sigma = max(sigma, (x_values[1] - x_values[0]) * 2)  # At least 2 data points wide
                
                sigmas[i] = sigma
                
            except Exception as e:
                print(f"    - Width estimation failed for peak {i}: {e}")
                # Fallback to simple spacing
                sigmas[i] = (x_values[-1] - x_values[0]) / (n_peaks * 6) * width_factor
        
        print(f"    - Estimated sigmas: {sigmas}")
        return sigmas

    def _width_from_half_max(self, x_values, y_values, peak_idx):
        """Estimate width based on Full Width at Half Maximum (FWHM)."""
        peak_height = y_values[peak_idx]
        half_max = peak_height / 2
        
        # Find left side of half max
        left_idx = peak_idx
        for i in range(peak_idx, max(0, peak_idx - len(y_values)//4), -1):
            if y_values[i] <= half_max:
                left_idx = i
                break
        
        # Find right side of half max
        right_idx = peak_idx
        for i in range(peak_idx, min(len(y_values), peak_idx + len(y_values)//4)):
            if y_values[i] <= half_max:
                right_idx = i
                break
        
        # Calculate FWHM and convert to sigma
        if right_idx > left_idx:
            fwhm = x_values[right_idx] - x_values[left_idx]
            sigma = fwhm / 2.354820045  # FWHM to sigma conversion for Gaussian
            return sigma
        else:
            # Fallback if FWHM couldn't be determined
            return (x_values[-1] - x_values[0]) / 20

    def _width_from_quarter_max(self, x_values, y_values, peak_idx):
        """Estimate width based on Full Width at Quarter Maximum (more conservative)."""
        peak_height = y_values[peak_idx]
        quarter_max = peak_height / 4
        
        # Find left side
        left_idx = peak_idx
        for i in range(peak_idx, max(0, peak_idx - len(y_values)//3), -1):
            if y_values[i] <= quarter_max:
                left_idx = i
                break
        
        # Find right side
        right_idx = peak_idx
        for i in range(peak_idx, min(len(y_values), peak_idx + len(y_values)//3)):
            if y_values[i] <= quarter_max:
                right_idx = i
                break
        
        if right_idx > left_idx:
            # Convert quarter-max width to approximate sigma
            quarter_width = x_values[right_idx] - x_values[left_idx]
            # For Gaussian, width at 1/4 max ≈ 3.33 * sigma
            sigma = quarter_width / 3.33
            return sigma
        else:
            return (x_values[-1] - x_values[0]) / 15

    def _width_from_spacing(self, x_values, detected_peaks, peak_index):
        """Estimate width based on spacing to neighboring peaks."""
        n_peaks = len(detected_peaks)
        
        if n_peaks == 1:
            # Single peak: use fraction of total range
            return (x_values[-1] - x_values[0]) / 8
        
        peak_positions = [x_values[idx] for idx in detected_peaks]
        current_pos = peak_positions[peak_index]
        
        # Find distances to neighbors
        distances = []
        for i, pos in enumerate(peak_positions):
            if i != peak_index:
                distances.append(abs(pos - current_pos))
        
        if distances:
            min_distance = min(distances)
            # Use fraction of minimum distance as sigma
            sigma = min_distance / 4  # Conservative: peak extends 1/4 way to nearest neighbor
            return sigma
        else:
            return (x_values[-1] - x_values[0]) / 10

    def _width_from_curvature(self, x_values, y_values, peak_idx):
        """Estimate width based on local curvature around the peak."""
        try:
            # Calculate second derivative around the peak
            window = min(5, len(y_values) // 10)  # Adaptive window size
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(y_values), peak_idx + window + 1)
            
            x_local = x_values[start_idx:end_idx]
            y_local = y_values[start_idx:end_idx]
            
            if len(x_local) < 3:
                return (x_values[-1] - x_values[0]) / 12
            
            # Calculate numerical second derivative
            dy = np.gradient(y_local, x_local)
            d2y = np.gradient(dy, x_local)
            
            # Find second derivative at peak (adjust index for local array)
            local_peak_idx = peak_idx - start_idx
            if 0 <= local_peak_idx < len(d2y):
                curvature = abs(d2y[local_peak_idx])
                
                # For Gaussian: sigma ≈ sqrt(amplitude / curvature)
                if curvature > 0:
                    amplitude = y_values[peak_idx]
                    sigma = np.sqrt(amplitude / curvature) * 0.1  # Scale factor
                    return max(sigma, (x_values[1] - x_values[0]))  # Minimum one data point
            
            # Fallback
            return (x_values[-1] - x_values[0]) / 15
            
        except Exception as e:
            return (x_values[-1] - x_values[0]) / 15

    def _width_adaptive(self, x_values, y_values, peak_idx, all_peaks, peak_index):
        """Adaptive width estimation combining multiple methods."""
        try:
            # Try multiple methods and use the median
            widths = []
            
            # Method 1: Half-max
            try:
                w1 = self._width_from_half_max(x_values, y_values, peak_idx)
                if w1 > 0:
                    widths.append(w1)
            except:
                pass
            
            # Method 2: Spacing-based
            try:
                w2 = self._width_from_spacing(x_values, all_peaks, peak_index)
                if w2 > 0:
                    widths.append(w2)
            except:
                pass
            
            # Method 3: Local variance estimate
            try:
                window = min(10, len(y_values) // 8)
                start_idx = max(0, peak_idx - window)
                end_idx = min(len(y_values), peak_idx + window + 1)
                
                x_local = x_values[start_idx:end_idx]
                y_local = y_values[start_idx:end_idx]
                
                if len(x_local) >= 3:
                    # Weight by intensity to get center of mass
                    total_intensity = np.sum(y_local)
                    if total_intensity > 0:
                        center_of_mass = np.sum(x_local * y_local) / total_intensity
                        # Calculate weighted variance
                        variance = np.sum(y_local * (x_local - center_of_mass)**2) / total_intensity
                        w3 = np.sqrt(variance)
                        if w3 > 0:
                            widths.append(w3)
            except:
                pass
            
            # Select median width if we have multiple estimates
            if len(widths) >= 2:
                return np.median(widths)
            elif len(widths) == 1:
                return widths[0]
            else:
                # Ultimate fallback
                return (x_values[-1] - x_values[0]) / (len(all_peaks) * 6)
        
        except Exception as e:
            return (x_values[-1] - x_values[0]) / 12


    def compare_with_reference(self, axis_name, normalize_method='area'):
        """
        Compare all individual profiles with the fitted reference.
        
        Parameters:
        -----------
        axis_name : str
            Axis identifier
        normalize_method : str
            How to normalize profiles: 'area', 'peak', 'none'
        
        Returns:
        --------
        dict : Comparison results for all profiles
        """
        if axis_name not in self.fitted_references:
            raise ValueError(f"No fitted reference found for {axis_name}. Run fit_reference_gaussian first.")
        
        ref_data = self.reference_profiles[axis_name]
        fitted_ref = self.fitted_references[axis_name]
        
        individual_profiles = ref_data['individual_profiles']
        x_axis = ref_data['x_axis']  # Original sparse x-axis
        
        # Create reference function for comparison - this uses the fitted Gaussian parameters
        # from the high-resolution upsampled average profile
        ref_func = lambda x: self.multi_gaussian(x, *self._flatten_parameters(fitted_ref['parameters']))
        
        comparison_results = []
        
        for i, profile in enumerate(individual_profiles):
            # CRITICAL: Evaluate the fitted reference function on the SAME x-axis as individual profiles
            # This ensures we're comparing the fitted Gaussian (from upsampled average) 
            # with each individual profile on their common original coordinate system
            ref_profile = ref_func(x_axis)
            
            # Store raw profiles BEFORE normalization for amplitude comparison
            profile_raw = profile.copy()
            ref_profile_raw = ref_profile.copy()
            
            # Debug: Ensure same lengths
            assert len(profile) == len(ref_profile) == len(x_axis), \
                f"Length mismatch: profile={len(profile)}, reference={len(ref_profile)}, x_axis={len(x_axis)}"
            
            # Normalize profiles if requested
            if normalize_method == 'area':
                profile_area = safe_trapezoid(profile, x_axis)
                ref_area = safe_trapezoid(ref_profile, x_axis)
                profile_norm = profile / profile_area if profile_area > 0 else profile
                ref_norm = ref_profile / ref_area if ref_area > 0 else ref_profile
            elif normalize_method == 'peak':
                profile_norm = profile / np.max(profile) if np.max(profile) > 0 else profile
                ref_norm = ref_profile / np.max(ref_profile) if np.max(ref_profile) > 0 else ref_profile
            else:  # 'none'
                profile_norm = profile
                ref_norm = ref_profile
            
            # Final check: ensure normalized profiles have same length
            assert len(profile_norm) == len(ref_norm), \
                f"Normalized profiles have different lengths: {len(profile_norm)} vs {len(ref_norm)}"
            
            # Calculate statistical shape descriptors comparing individual profile 
            # with fitted reference (from upsampled average)
            # Pass both normalized and raw profiles for proper AUC comparison
            shape_comparison = self._calculate_shape_comparison(
                x_axis, profile_norm, ref_norm, 
                profile_raw=profile_raw, reference_raw=ref_profile_raw
            )
            
            comparison_results.append({
                'profile_index': i,
                'shape_descriptors': shape_comparison,
            })
        
        self.comparison_results[axis_name] = comparison_results
        return comparison_results
    
    def _flatten_parameters(self, parameters):
        """Convert parameter list to flat array for multi_gaussian."""
        flat = []
        for param in parameters:
            flat.extend([param['amplitude'], param['center'], param['sigma']])
        return flat
    
    def _calculate_shape_comparison(self, x_axis, profile, reference, profile_raw=None, reference_raw=None):
        """Calculate statistical shape descriptors comparing profile to reference."""
        # Use profile_raw and reference_raw for AUC calculations if provided
        if profile_raw is not None and reference_raw is not None:
            auc_comparison = self._calculate_auc_comparison(x_axis, profile_raw, reference_raw)
        else:
            auc_comparison = self._calculate_auc_comparison(x_axis, profile, reference)
        
        # Ensure both profiles are valid with more robust checking
        profile_sum = np.sum(profile)
        reference_sum = np.sum(reference)
        
        if profile_sum <= 1e-15 or reference_sum <= 1e-15 or np.any(np.isnan(profile)) or np.any(np.isnan(reference)):
            return self._empty_shape_descriptors()
        
        # Normalize to probability distributions with robustness checks
        profile_prob = profile / profile_sum
        reference_prob = reference / reference_sum
        
        # Additional validation after normalization
        if np.any(np.isnan(profile_prob)) or np.any(np.isnan(reference_prob)):
            return self._empty_shape_descriptors()
        
        # Calculate moments
        profile_moments = self._calculate_moments(x_axis, profile_prob)
        reference_moments = self._calculate_moments(x_axis, reference_prob)
        
        # Moment differences
        moment_differences = {
            'centroid_diff': profile_moments['centroid'] - reference_moments['centroid'],
            'width_ratio': profile_moments['std_dev'] / reference_moments['std_dev'] if reference_moments['std_dev'] > 0 else np.nan,
            'skewness_diff': profile_moments['skewness'] - reference_moments['skewness'],
            'kurtosis_diff': profile_moments['kurtosis'] - reference_moments['kurtosis']
        }
        
        # Similarity measures
        correlation = np.corrcoef(profile, reference)[0, 1] if np.std(profile) > 0 and np.std(reference) > 0 else np.nan
        
        # Cosine similarity
        cosine_similarity = self._calculate_cosine_similarity(profile, reference)
        
        # Euclidean distance using raw data
        if profile_raw is not None and reference_raw is not None:
            euclidean_distance = self._calculate_euclidean_distance(profile_raw, reference_raw)
        else:
            euclidean_distance = self._calculate_euclidean_distance(profile, reference)
        
        # Overlap integral
        overlap_integral = safe_trapezoid(np.minimum(profile_prob, reference_prob), x_axis)
        
        # Chi-squared goodness of fit
        chi_squared = np.sum((profile - reference)**2 / (reference + 1e-10))
        
        # Kolmogorov-Smirnov statistic
        ks_statistic = self._ks_statistic(x_axis, profile_prob, reference_prob)
        
        return {
            'profile_moments': profile_moments,
            'reference_moments': reference_moments,
            'moment_differences': moment_differences,
            'auc_comparison': auc_comparison,
            'similarity_measures': {
                'correlation': correlation,
                'cosine_similarity': cosine_similarity,
                'euclidean_distance': euclidean_distance,
                'overlap_integral': overlap_integral,
                'chi_squared': chi_squared,
                'ks_statistic': ks_statistic
            }
        }
    
    def _calculate_auc_comparison(self, x_axis, profile, reference, noise_threshold_factor=1000):
        """
        Calculate Area Under Curve comparison with noise filtering.
        
        Parameters:
        -----------
        x_axis : numpy.ndarray
            X-axis coordinates
        profile : numpy.ndarray
            Individual profile intensities
        reference : numpy.ndarray
            Reference profile intensities
        noise_threshold_factor : float
            Factor for noise filtering (values < max/factor are set to 0)
        
        Returns:
        --------
        dict : AUC comparison metrics
        """
        
        # Apply noise filtering
        profile_filtered = self._apply_noise_filter(profile, noise_threshold_factor)
        reference_filtered = self._apply_noise_filter(reference, noise_threshold_factor)
        
        # Calculate AUC for filtered profiles
        profile_auc = safe_trapezoid(profile_filtered, x_axis)
        reference_auc = safe_trapezoid(reference_filtered, x_axis)
        
        # Calculate AUC ratio
        auc_ratio = profile_auc / reference_auc if reference_auc > 0 else np.nan
        
        # Calculate absolute AUC difference
        auc_difference = profile_auc - reference_auc
        
        # Calculate relative AUC difference (percentage)
        auc_relative_diff = (auc_difference / reference_auc * 100) if reference_auc > 0 else np.nan
        
        # Calculate effective signal region (number of points above noise threshold)
        profile_signal_points = np.sum(profile_filtered > 0)
        reference_signal_points = np.sum(reference_filtered > 0)
        
        return {
            'profile_auc': profile_auc,
            'reference_auc': reference_auc,
            'auc_ratio': auc_ratio,
            'auc_difference': auc_difference,
            'auc_relative_diff_percent': auc_relative_diff,
            'profile_signal_points': profile_signal_points,
            'reference_signal_points': reference_signal_points,
            'noise_threshold_factor': noise_threshold_factor
        }
    
    def _apply_noise_filter(self, profile, noise_threshold_factor):
        """
        Apply noise filtering by setting values below max/threshold_factor to zero.
        
        Parameters:
        -----------
        profile : numpy.ndarray
            Input intensity profile
        noise_threshold_factor : float
            Threshold factor (values < max/factor are filtered out)
        
        Returns:
        --------
        numpy.ndarray : Filtered profile
        """
        if np.max(profile) <= 0:
            return profile.copy()
        
        threshold = np.max(profile) / noise_threshold_factor
        filtered_profile = profile.copy()
        filtered_profile[filtered_profile < threshold] = 0
        
        return filtered_profile
    
    def _calculate_moments(self, x_axis, prob_distribution):
        """Calculate statistical moments of a probability distribution."""
        # Add robustness checks
        if np.sum(prob_distribution) <= 1e-15:
            return {
                'centroid': np.nan,
                'variance': np.nan,
                'std_dev': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan
            }
        
        # Ensure proper normalization
        prob_norm = prob_distribution / np.sum(prob_distribution)
        
        # Calculate centroid (first moment)
        centroid = np.sum(x_axis * prob_norm)
        
        # Calculate variance (second central moment)
        variance = np.sum(prob_norm * (x_axis - centroid)**2)
        std_dev = np.sqrt(variance)
        
        # Calculate higher moments with robustness checks
        if std_dev > 1e-10:  # More robust threshold
            standardized = (x_axis - centroid) / std_dev
            skewness = np.sum(prob_norm * standardized**3)
            kurtosis = np.sum(prob_norm * standardized**4)
        else:
            skewness = np.nan  # More appropriate than 0 for undefined case
            kurtosis = np.nan
        
        return {
            'centroid': centroid,
            'variance': variance,
            'std_dev': std_dev,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _calculate_cosine_similarity(self, profile, reference):
        """
        Calculate cosine similarity between two profiles.
        
        Cosine similarity = (A · B) / (||A|| × ||B||)
        where A and B are the profile vectors.
        
        Returns value between -1 and 1:
        - 1.0: Identical direction (perfect similarity)
        - 0.0: Orthogonal (no similarity)
        - -1.0: Opposite direction (perfect dissimilarity)
        """
        
        # Handle edge cases
        if len(profile) != len(reference):
            return np.nan
        
        # Calculate norms
        norm_profile = np.linalg.norm(profile)
        norm_reference = np.linalg.norm(reference)
        
        # Avoid division by zero
        if norm_profile == 0 or norm_reference == 0:
            return np.nan
        
        # Calculate dot product
        dot_product = np.dot(profile, reference)
        
        # Calculate cosine similarity
        cosine_sim = dot_product / (norm_profile * norm_reference)
        
        return cosine_sim
    
    def _calculate_euclidean_distance(self, profile, reference):
        """
        Calculate Euclidean distance between two profiles.
        
        Euclidean distance = sqrt(sum((A - B)^2))
        where A and B are the profile vectors.
        
        Returns:
        --------
        float : Euclidean distance (always non-negative)
                Lower values indicate higher similarity
        """
        
        # Handle edge cases
        if len(profile) != len(reference):
            return np.nan
        
        # Calculate squared differences
        squared_diffs = (profile - reference) ** 2
        
        # Calculate Euclidean distance
        euclidean_dist = np.sqrt(np.sum(squared_diffs))
        
        return euclidean_dist
    
    def _ks_statistic(self, x_axis, prob1, prob2):
        """Calculate Kolmogorov-Smirnov statistic."""
        cumsum1 = np.cumsum(prob1)
        cumsum2 = np.cumsum(prob2)
        return np.max(np.abs(cumsum1 - cumsum2))
    
    def _empty_shape_descriptors(self):
        """Return empty shape descriptors for invalid profiles."""
        return {
            'profile_moments': {'centroid': np.nan, 'variance': np.nan, 'std_dev': np.nan, 'skewness': np.nan, 'kurtosis': np.nan},
            'reference_moments': {'centroid': np.nan, 'variance': np.nan, 'std_dev': np.nan, 'skewness': np.nan, 'kurtosis': np.nan},
            'moment_differences': {'centroid_diff': np.nan, 'width_ratio': np.nan, 'skewness_diff': np.nan, 'kurtosis_diff': np.nan},
            'auc_comparison': {
                'profile_auc': np.nan, 'reference_auc': np.nan, 'auc_ratio': np.nan, 
                'auc_difference': np.nan, 'auc_relative_diff_percent': np.nan,
                'profile_signal_points': 0, 'reference_signal_points': 0, 'noise_threshold_factor': 1000
            },
            'similarity_measures': {'correlation': np.nan, 'cosine_similarity': np.nan, 'euclidean_distance': np.nan, 'overlap_integral': np.nan, 'chi_squared': np.nan, 'ks_statistic': np.nan}
        }

    def get_analysis_summary(self, axis_name):
        """Get summary statistics for the comparison analysis."""
        if axis_name not in self.comparison_results:
            return None
        
        results = self.comparison_results[axis_name]
        
        # Collect all valid measurements
        width_ratios = []
        correlations = []
        cosine_similarities = []
        euclidean_distances = []
        centroid_diffs = []
        skewness_diffs = []
        kurtosis_diffs = []
        auc_ratios = []
        auc_relative_diffs = []
        
        for result in results:
            descriptors = result['shape_descriptors']
            
            if not np.isnan(descriptors['moment_differences']['width_ratio']):
                width_ratios.append(descriptors['moment_differences']['width_ratio'])
            
            if not np.isnan(descriptors['similarity_measures']['correlation']):
                correlations.append(descriptors['similarity_measures']['correlation'])
            
            if not np.isnan(descriptors['similarity_measures']['cosine_similarity']):
                cosine_similarities.append(descriptors['similarity_measures']['cosine_similarity'])
            
            if not np.isnan(descriptors['similarity_measures']['euclidean_distance']):
                euclidean_distances.append(descriptors['similarity_measures']['euclidean_distance'])
            
            if not np.isnan(descriptors['moment_differences']['centroid_diff']):
                centroid_diffs.append(descriptors['moment_differences']['centroid_diff'])
            
            if not np.isnan(descriptors['moment_differences']['skewness_diff']):
                skewness_diffs.append(descriptors['moment_differences']['skewness_diff'])
            
            if not np.isnan(descriptors['moment_differences']['kurtosis_diff']):
                kurtosis_diffs.append(descriptors['moment_differences']['kurtosis_diff'])
            
            # AUC metrics
            if 'auc_comparison' in descriptors:
                auc_comp = descriptors['auc_comparison']
                if not np.isnan(auc_comp['auc_ratio']):
                    auc_ratios.append(auc_comp['auc_ratio'])
                if not np.isnan(auc_comp['auc_relative_diff_percent']):
                    auc_relative_diffs.append(auc_comp['auc_relative_diff_percent'])
        
        summary = {
            'n_profiles': len(results),
            'width_ratio_stats': {
                'mean': np.mean(width_ratios) if width_ratios else np.nan,
                'std': np.std(width_ratios) if width_ratios else np.nan,
                'median': np.median(width_ratios) if width_ratios else np.nan
            },
            'correlation_stats': {
                'mean': np.mean(correlations) if correlations else np.nan,
                'std': np.std(correlations) if correlations else np.nan,
                'median': np.median(correlations) if correlations else np.nan
            },
            'cosine_similarity_stats': {
                'mean': np.mean(cosine_similarities) if cosine_similarities else np.nan,
                'std': np.std(cosine_similarities) if cosine_similarities else np.nan,
                'median': np.median(cosine_similarities) if cosine_similarities else np.nan
            },
            'euclidean_distance_stats': {
                'mean': np.mean(euclidean_distances) if euclidean_distances else np.nan,
                'std': np.std(euclidean_distances) if euclidean_distances else np.nan,
                'median': np.median(euclidean_distances) if euclidean_distances else np.nan
            },
            'centroid_shift_stats': {
                'mean': np.mean(centroid_diffs) if centroid_diffs else np.nan,
                'std': np.std(centroid_diffs) if centroid_diffs else np.nan,
                'median': np.median(centroid_diffs) if centroid_diffs else np.nan
            },
            'skewness_variation_stats': {
                'mean': np.mean(skewness_diffs) if skewness_diffs else np.nan,
                'std': np.std(skewness_diffs) if skewness_diffs else np.nan,
                'median': np.median(skewness_diffs) if skewness_diffs else np.nan
            },
            'kurtosis_variation_stats': {
                'mean': np.mean(kurtosis_diffs) if kurtosis_diffs else np.nan,
                'std': np.std(kurtosis_diffs) if kurtosis_diffs else np.nan,
                'median': np.median(kurtosis_diffs) if kurtosis_diffs else np.nan
            },
            'auc_ratio_stats': {
                'mean': np.mean(auc_ratios) if auc_ratios else np.nan,
                'std': np.std(auc_ratios) if auc_ratios else np.nan,
                'median': np.median(auc_ratios) if auc_ratios else np.nan
            },
            'auc_relative_diff_stats': {
                'mean': np.mean(auc_relative_diffs) if auc_relative_diffs else np.nan,
                'std': np.std(auc_relative_diffs) if auc_relative_diffs else np.nan,
                'median': np.median(auc_relative_diffs) if auc_relative_diffs else np.nan
            }
        }
        
        return summary


def analyze_peak_shapes_with_reference(data, X_Axis1, X_Axis2, X_Axis3, 
                                     high_res_factor=10, max_peaks=3, 
                                     manual_peaks=None):
    """
    Complete analysis workflow using reference-based comparison with GMM fitting.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data with shape (NoPoints, NoAxis1, NoAxis2, NoAxis3)
    X_Axis1, X_Axis2, X_Axis3 : numpy.ndarray
        Coordinate arrays for each axis
    high_res_factor : int
        Resolution enhancement factor for reference fitting
    max_peaks : int
        Maximum number of peaks to fit in reference
    manual_peaks : dict or None
        Manual peak positions for each axis, e.g., 
        {'axis1': [pos1, pos2], 'axis2': [pos1], 'axis3': [pos1, pos2, pos3]}
    
    Returns:
    --------
    dict : Complete analysis results
    """
    
    analyzer = ReferenceBasedPeakComparison(high_res_factor=high_res_factor)
    
    results = {
        'analyzer': analyzer,
        'reference_profiles': {},
        'fitted_references': {},
        'comparison_results': {},
        'summaries': {}
    }
    
    axes_info = [
        ('axis1', X_Axis1),
        ('axis2', X_Axis2), 
        ('axis3', X_Axis3)
    ]
    
    for axis_name, x_axis in axes_info:
        print(f"Processing {axis_name}...")
        
        # Step 1: Create averaged reference
        ref_profile = analyzer.create_averaged_reference(data, x_axis, axis_name)
        results['reference_profiles'][axis_name] = ref_profile
        
        # Step 2: Get manual peak positions for this axis if provided
        manual_peaks_for_axis = None
        if manual_peaks is not None and axis_name in manual_peaks:
            manual_peaks_for_axis = manual_peaks[axis_name]
            print(f"  - Using manual peaks: {manual_peaks_for_axis}")
        
        # Step 3: Fit Gaussian reference using GMM
        fit_result = analyzer.fit_reference_gaussian(
            axis_name, 
            max_peaks=max_peaks,
            manual_peak_positions=manual_peaks_for_axis
        )
        results['fitted_references'][axis_name] = fit_result
        
        if fit_result['success']:
            print(f"  - Successfully fitted {analyzer.fitted_references[axis_name]['n_peaks']} peaks")
            print(f"  - R-squared: {fit_result['r_squared']:.4f}")
            print(f"  - Method: {fit_result.get('method', 'Unknown')}")
            
            # Step 4: Compare all profiles with reference
            comparison = analyzer.compare_with_reference(axis_name)
            results['comparison_results'][axis_name] = comparison
            
            # Step 5: Generate summary
            summary = analyzer.get_analysis_summary(axis_name)
            results['summaries'][axis_name] = summary
            
            print(f"  - Mean correlation with reference: {summary['correlation_stats']['mean']:.4f}")
            print(f"  - Mean width ratio: {summary['width_ratio_stats']['mean']:.4f}")
            print(f"  - Mean Euclidean distance: {summary['euclidean_distance_stats']['mean']:.4f}")
        else:
            print(f"  - Fitting failed: {fit_result.get('error', 'Unknown error')}")
    
    return results


# Example usage
if __name__ == "__main__":
    # Create example data
    NoPoints, NoAxis1, NoAxis2, NoAxis3 = 50, 30, 25, 20
    
    # Simulate data with varying peak shapes
    np.random.seed(42)
    data = np.random.rand(NoPoints, NoAxis1, NoAxis2, NoAxis3)
    
    # Add some structure (peaks) to the data
    X_Axis1 = np.linspace(-5, 5, NoAxis1)
    X_Axis2 = np.linspace(-3, 3, NoAxis2)
    X_Axis3 = np.linspace(-2, 2, NoAxis3)
    
    # Example with manual peak positions for problematic fitting
    manual_peaks = {
        'axis1': [-2.0, 1.5],  # Two peaks in axis1
        'axis2': [0.0],        # Single peak in axis2
        'axis3': [-1.0, 0.5, 1.2]  # Three peaks in axis3
    }
    
    # Run complete analysis with manual peak suggestions
    analysis_results = analyze_peak_shapes_with_reference(
        data, X_Axis1, X_Axis2, X_Axis3,
        high_res_factor=5, max_peaks=3,
        manual_peaks=manual_peaks
    )
    
    print(f"\n=== Analysis Summary ===")
    for axis_name, summary in analysis_results['summaries'].items():
        if summary:
            print(f"\n{axis_name.upper()}:")
            print(f"  Profiles analyzed: {summary['n_profiles']}")
            print(f"  Average correlation: {summary['correlation_stats']['mean']:.4f} ± {summary['correlation_stats']['std']:.4f}")
            print(f"  Average width ratio: {summary['width_ratio_stats']['mean']:.4f} ± {summary['width_ratio_stats']['std']:.4f}")
            print(f"  Average Euclidean distance: {summary['euclidean_distance_stats']['mean']:.4f} ± {summary['euclidean_distance_stats']['std']:.4f}")
    
    # Alternative: Run without manual peaks (automatic detection)
    print(f"\n=== Running with Automatic Peak Detection ===")
    analysis_results_auto = analyze_peak_shapes_with_reference(
        data, X_Axis1, X_Axis2, X_Axis3,
        high_res_factor=5, max_peaks=2
    )


"""# for visualization"""
import matplotlib.pyplot as plt
import numpy as np


def visualize_reference_analysis(analyzer, axis_name, figsize=(15, 10)):
    """
    Comprehensive visualization of reference-based peak analysis.
    
    Parameters:
    -----------
    analyzer : ReferenceBasedPeakComparison
        The analyzer object after running the analysis
    axis_name : str
        Axis name ('axis1', 'axis2', or 'axis3')
    figsize : tuple
        Figure size (width, height)
    """
    
    if axis_name not in analyzer.reference_profiles:
        print(f"No reference data found for {axis_name}")
        return
    
    ref_data = analyzer.reference_profiles[axis_name]
    x_axis = ref_data['x_axis']
    avg_profile = ref_data['average_profile']
    std_profile = ref_data['std_profile']
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Reference-Based Peak Analysis: {axis_name.upper()}'
                 , x = 0.5, y = 0.85
                 , fontsize=16, fontweight='bold')
    
    # Plot 1: Reference profile with fitted Gaussian
    ax1 = axes[0, 0]
    ax1.plot(x_axis, avg_profile, 'b-', linewidth=2, label='Averaged Reference', alpha=0.8)
    ax1.fill_between(x_axis, avg_profile - std_profile, avg_profile + std_profile, 
                     alpha=0.3, color='blue', label='±1σ variation')
    
    # Add fitted Gaussian if available
    if axis_name in analyzer.fitted_references and analyzer.fitted_references[axis_name]:
        fitted_ref = analyzer.fitted_references[axis_name]
        if 'x_high_res' in fitted_ref and 'fitted_curve' in fitted_ref:
            x_high_res = fitted_ref['x_high_res']
            fitted_curve = fitted_ref['fitted_curve']
            ax1.plot(x_high_res, fitted_curve, 'r--', linewidth=2, label='Fitted Gaussian(s)', alpha=0.9)
            
            # Add R-squared info
            r_squared = fitted_ref.get('r_squared', 0)
            ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=10)
    
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Reference Profile vs Fitted Gaussian')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual profiles comparison
    ax2 = axes[0, 1]
    individual_profiles = ref_data['individual_profiles']
    n_profiles = min(10, len(individual_profiles))  # Show max 10 profiles
    
    # randomly select several samples
    selected_indices = np.random.choice(len(individual_profiles), size=n_profiles, replace=False)
    selected_indices_sorted = np.sort(selected_indices)
    individual_profiles_selected = individual_profiles[selected_indices_sorted]

    # Plot a few individual profiles
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    for i in range(n_profiles):
        profile = individual_profiles_selected[i]
        ax2.plot(x_axis, profile, color=colors[i], alpha=0.6, linewidth=1)
    
    # Overlay the reference
    ax2.plot(x_axis, avg_profile, 'r-', linewidth=3, label='Average Reference', alpha=0.9)
    
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Intensity')
    ax2.set_title(f'Individual Profiles (showing {n_profiles}/{len(individual_profiles)})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison metrics distribution
    ax3 = axes[1, 0]
    if axis_name in analyzer.comparison_results:
        results = analyzer.comparison_results[axis_name]
        
        # Extract metrics
        width_ratios = []
        correlations = []
        centroid_shifts = []
        
        for result in results:
            descriptors = result['shape_descriptors']
            
            wr = descriptors['moment_differences']['width_ratio']
            if not np.isnan(wr):
                width_ratios.append(wr)
            
            corr = descriptors['similarity_measures']['correlation']
            if not np.isnan(corr):
                correlations.append(corr)
            
            shift = descriptors['moment_differences']['centroid_diff']
            if not np.isnan(shift):
                centroid_shifts.append(shift)
        
        # Create histograms
        if width_ratios:
            ax3.hist(width_ratios, bins=20, alpha=0.7, color='blue', label=f'Width Ratio\n(μ={np.mean(width_ratios):.3f})')
        if correlations:
            ax3_twin = ax3.twinx()
            ax3_twin.hist(correlations, bins=20, alpha=0.7, color='red', label=f'Correlation\n(μ={np.mean(correlations):.3f})')
            ax3_twin.set_ylabel('Correlation Count', color='red')
            ax3_twin.tick_params(axis='y', labelcolor='red')
        
        ax3.set_xlabel('Width Ratio')
        ax3.set_ylabel('Width Ratio Count', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')
        ax3.set_title('Distribution of Comparison Metrics')
        ax3.legend(loc='upper left')
        if correlations:
            ax3_twin.legend(loc='upper right')
    
    # Plot 4: Example comparison - best and worst matches
    ax4 = axes[1, 1]
    if axis_name in analyzer.comparison_results:
        results = analyzer.comparison_results[axis_name]
        
        # Find best and worst correlations
        correlations_with_idx = []
        for i, result in enumerate(results):
            corr = result['shape_descriptors']['similarity_measures']['correlation']
            if not np.isnan(corr):
                correlations_with_idx.append((corr, i))
        
        if correlations_with_idx:
            correlations_with_idx.sort(key=lambda x: x[0], reverse=True)
            
            # Best match
            best_corr, best_idx = correlations_with_idx[0]
            best_profile = individual_profiles[best_idx]
            
            # Worst match (if more than one profile)
            if len(correlations_with_idx) > 1:
                worst_corr, worst_idx = correlations_with_idx[-1]
                worst_profile = individual_profiles[worst_idx]
            else:
                worst_corr, worst_idx = best_corr, best_idx
                worst_profile = best_profile
            
            # Plot comparison
            ax4.plot(x_axis, avg_profile, 'k-', linewidth=2, label='Reference', alpha=0.8)
            ax4.plot(x_axis, best_profile, 'g--', linewidth=2, label=f'Best Match (r={best_corr:.3f})', alpha=0.8)
            
            if best_idx != worst_idx:
                ax4.plot(x_axis, worst_profile, 'r:', linewidth=2, label=f'Worst Match (r={worst_corr:.3f})', alpha=0.8)
    
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Best vs Worst Profile Matches')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_reference_analysis_simple(analyzer, axis_name, figsize=(15, 10),
                                        title = None, labels = None, X_title = None):
    """
    Comprehensive visualization of reference-based peak analysis, but a simple one with less informatin.
    
    Parameters:
    -----------
    analyzer : ReferenceBasedPeakComparison
        The analyzer object after running the analysis
    axis_name : str
        Axis name ('axis1', 'axis2', or 'axis3')
    figsize : tuple
        Figure size (width, height)
    """
    
    if axis_name not in analyzer.reference_profiles:
        print(f"No reference data found for {axis_name}")
        return
    
    ref_data = analyzer.reference_profiles[axis_name]
    x_axis = ref_data['x_axis']
    avg_profile = ref_data['average_profile']
    std_profile = ref_data['std_profile']
    
    # Create subplot layout
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi = 300)
    if title:
        fig.suptitle(f'Reference-Based Peak Analysis: {axis_name.upper()}'
                    , x = 0.5, y = 0.85
                    , fontsize=16, fontweight='bold')
    
    # Plot 1: Reference profile with fitted Gaussian
    ax1 = axes[0]
    ax1.plot(x_axis, avg_profile, 'b-', linewidth=2, label='Averaged Reference', alpha=0.8)
    ax1.fill_between(x_axis, avg_profile - std_profile, avg_profile + std_profile, 
                     alpha=0.3, color='blue', label='±1σ variation')
    
    # Add fitted Gaussian if available
    if axis_name in analyzer.fitted_references and analyzer.fitted_references[axis_name]:
        fitted_ref = analyzer.fitted_references[axis_name]
        if 'x_high_res' in fitted_ref and 'fitted_curve' in fitted_ref:
            x_high_res = fitted_ref['x_high_res']
            fitted_curve = fitted_ref['fitted_curve']
            ax1.plot(x_high_res, fitted_curve, 'r--', linewidth=2, label='Fitted Gaussian', alpha=0.9)
            
            # Add R-squared info
            r_squared = fitted_ref.get('r_squared', 0)
            ax1.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax1.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=14)
    
    ax1.set_ylabel('Intensity [a.u.]')
    ax1.set_title('Reference Profile vs Fitted Gaussian')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Individual profiles comparison
    ax2 = axes[1]
    individual_profiles = ref_data['individual_profiles']
    n_profiles = min(10, len(individual_profiles))  # Show max 10 profiles
    
    # randomly select several samples
    selected_indices = np.random.choice(len(individual_profiles), size=n_profiles, replace=False)
    selected_indices_sorted = np.sort(selected_indices)
    individual_profiles_selected = individual_profiles[selected_indices_sorted]

    # Plot a few individual profiles
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    for i in range(n_profiles):
        profile = individual_profiles_selected[i]
        ax2.plot(x_axis, profile, color=colors[i], alpha=0.6, linewidth=1)
    
    # Overlay the reference
    ax2.plot(x_axis, avg_profile, 'ro-', linewidth=1, label='Average Reference', alpha=0.9)
    
    ax2.set_ylabel('Intensity [a.u.]')
    ax2.set_title(f'Individual Profiles (showing {n_profiles}/{len(individual_profiles)})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if labels:
        for ax, label in zip(axes, labels):
            ax.text(-0.02, 1.02, f'({label})', transform=ax.transAxes, fontsize=26, fontweight='bold', color='black',
            bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    if X_title:
        for ax in axes:
            ax.set_xlabel(X_title)
    else:
        ax.set_xlabel('Position')
    
    plt.tight_layout()
    return fig


def visualize_specific_profile_comparison(analyzer, axis_name, profile_index, figsize=(15, 10)):
    """
    Detailed comparison of a specific profile against the reference.
    
    Parameters:
    -----------
    analyzer : ReferenceBasedPeakComparison
        The analyzer object
    axis_name : str
        Axis name
    profile_index : int
        Index of the specific profile to analyze
    figsize : tuple
        Figure size
    """
    
    if axis_name not in analyzer.reference_profiles:
        print(f"No reference data found for {axis_name}")
        return
    
    ref_data = analyzer.reference_profiles[axis_name]
    x_axis = ref_data['x_axis']
    avg_profile = ref_data['average_profile']
    individual_profiles = ref_data['individual_profiles']
    
    if profile_index >= len(individual_profiles):
        print(f"Profile index {profile_index} out of range. Max index: {len(individual_profiles)-1}")
        return
    
    selected_profile = individual_profiles[profile_index]
    
    # Get fitted reference function
    if axis_name in analyzer.fitted_references:
        fitted_ref = analyzer.fitted_references[axis_name]
        ref_func = lambda x: analyzer.multi_gaussian(x, *analyzer._flatten_parameters(fitted_ref['parameters']))
        fitted_ref_profile = ref_func(x_axis)
    else:
        fitted_ref_profile = avg_profile
    
    # Get comparison metrics
    if axis_name in analyzer.comparison_results and profile_index < len(analyzer.comparison_results[axis_name]):
        metrics = analyzer.comparison_results[axis_name][profile_index]['shape_descriptors']
        width_ratio = metrics['moment_differences']['width_ratio']
        correlation = metrics['similarity_measures']['correlation']
        centroid_shift = metrics['moment_differences']['centroid_diff']
        skewness_change = metrics['moment_differences']['skewness_diff']
    else:
        width_ratio = correlation = centroid_shift = skewness_change = np.nan
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Detailed Profile Comparison: {axis_name.upper()}, Profile #{profile_index}', 
                 x = 0.5, y=0.85,  
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Direct profile comparison
    ax1 = axes[0, 0]
    ax1.plot(x_axis, fitted_ref_profile, 'b-', linewidth=2, label='Reference (fitted)', alpha=0.8)
    ax1.plot(x_axis, selected_profile, 'r--', linewidth=2, label=f'Profile #{profile_index}', alpha=0.8)
    
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Profile Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation text
    if not np.isnan(correlation):
        ax1.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize = 11)
    
    # Plot 2: Normalized comparison
    ax2 = axes[0, 1]
    
    # Normalize both profiles
    ref_norm = fitted_ref_profile / np.max(fitted_ref_profile) if np.max(fitted_ref_profile) > 0 else fitted_ref_profile
    prof_norm = selected_profile / np.max(selected_profile) if np.max(selected_profile) > 0 else selected_profile
    
    ax2.plot(x_axis, ref_norm, 'b-', linewidth=2, label='Reference (normalized)', alpha=0.8)
    ax2.plot(x_axis, prof_norm, 'r--', linewidth=2, label='Profile (normalized)', alpha=0.8)
    
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_title('Normalized Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual analysis
    ax3 = axes[1, 0]
    residual = selected_profile - fitted_ref_profile
    ax3.plot(x_axis, residual, 'g-', linewidth=2, alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.fill_between(x_axis, residual, alpha=0.3, color='green')
    
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Residual (Profile - Reference)')
    ax3.set_title('Residual Analysis')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')  # Turn off axis
    
    # Create two-column text summary
    # Left column
    width_interp = ('Same width' if abs(width_ratio - 1) < 0.1 else 
                   'Broader' if width_ratio > 1.1 else 
                   'Sharper' if not np.isnan(width_ratio) else 'N/A')
    
    corr_interp = ('Excellent match' if correlation > 0.95 else 
                  'Good match' if correlation > 0.8 else 
                  'Poor match' if not np.isnan(correlation) else 'N/A')
    
    left_column = f"""COMPARISON METRICS
==================

Width Ratio:      {width_ratio:.4f}
  → {width_interp}

Correlation:      {correlation:.4f}
  → {corr_interp}"""
    
    # Right column
    shift_interp = ('No shift' if abs(centroid_shift) < 0.1 else 
                   'Right shift' if centroid_shift > 0 else 
                   'Left shift' if not np.isnan(centroid_shift) else 'N/A')
    
    skew_interp = ('No asymmetry change' if abs(skewness_change) < 0.2 else 
                  'More right-tailed' if skewness_change > 0 else 
                  'More left-tailed' if not np.isnan(skewness_change) else 'N/A')
    
    right_column = f"""SHAPE ANALYSIS
==============

Centroid Shift:   {centroid_shift:+.4f}
  → {shift_interp}

Skewness Change:  {skewness_change:+.4f}
  → {skew_interp}"""
    
    # Display left column
    ax4.text(-0.05, 0.95, left_column, transform=ax4.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Display right column
    ax4.text(0.42, 0.95, right_column, transform=ax4.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    return fig


def quick_reference_overview(analyzer, axis_name):
    """
    Quick overview of the reference analysis with key statistics.
    """
    
    print("=" * 50)
    print(f"REFERENCE ANALYSIS OVERVIEW: {axis_name.upper()}")
    print("=" * 50)
    
    if axis_name not in analyzer.reference_profiles:
        print("No reference data available!")
        return
    
    ref_data = analyzer.reference_profiles[axis_name]
    n_profiles = len(ref_data['individual_profiles'])
    print(f"Number of individual profiles: {n_profiles}")
    
    # Reference fitting info
    if axis_name in analyzer.fitted_references:
        fitted_ref = analyzer.fitted_references[axis_name]
        if fitted_ref and 'success' in fitted_ref and fitted_ref.get('success', False):
            print(f"Gaussian fitting: SUCCESS")
            print(f"Number of peaks fitted: {fitted_ref.get('n_peaks', 'unknown')}")
            print(f"R-squared: {fitted_ref.get('r_squared', 'unknown'):.4f}")
            
            # Peak parameters
            if 'parameters' in fitted_ref:
                print(f"\nFitted Peak Parameters:")
                for i, param in enumerate(fitted_ref['parameters']):
                    print(f"  Peak {i+1}: Center={param['center']:.3f}, "
                          f"Amplitude={param['amplitude']:.3f}, "
                          f"Sigma={param['sigma']:.3f}")
        else:
            print(f"Gaussian fitting: FAILED")
            if 'error' in fitted_ref:
                print(f"Error: {fitted_ref['error']}")
    
    # Summary statistics
    if axis_name in analyzer.comparison_results:
        summary = analyzer.get_analysis_summary(axis_name)
        if summary:
            print(f"\nCOMPARISON SUMMARY:")
            print(f"Average correlation: {summary['correlation_stats']['mean']:.4f} ± {summary['correlation_stats']['std']:.4f}")
            print(f"Average width ratio: {summary['width_ratio_stats']['mean']:.4f} ± {summary['width_ratio_stats']['std']:.4f}")
            print(f"Average centroid shift: {summary['centroid_shift_stats']['mean']:+.4f} ± {summary['centroid_shift_stats']['std']:.4f}")
            print(f"Average skewness change: {summary['skewness_variation_stats']['mean']:+.4f} ± {summary['skewness_variation_stats']['std']:.4f}")


# Example usage function
def example_visualization_workflow(analyzer, axis_name='axis1'):
    """
    Example of how to use all visualization functions together.
    """
    
    print("Running comprehensive visualization workflow...")
    
    # 1. Quick overview
    quick_reference_overview(analyzer, axis_name)
    
    # 2. Comprehensive analysis visualization
    fig1 = visualize_reference_analysis(analyzer, axis_name)
    if fig1:
        plt.show()
    
    # 3. Specific profile comparisons
    if axis_name in analyzer.reference_profiles:
        n_profiles = len(analyzer.reference_profiles[axis_name]['individual_profiles'])
        
        # Show first profile
        print(f"\nShowing detailed comparison for Profile #0:")
        fig2 = visualize_specific_profile_comparison(analyzer, axis_name, 0)
        if fig2:
            plt.show()
        
        # Show a middle profile
        if n_profiles > 2:
            mid_idx = n_profiles // 2
            print(f"\nShowing detailed comparison for Profile #{mid_idx}:")
            fig3 = visualize_specific_profile_comparison(analyzer, axis_name, mid_idx)
            if fig3:
                plt.show()
    
    print("Visualization workflow complete!")


# if __name__ == "__main__":
#     print("Visualization functions loaded!")
#     print("Use: visualize_reference_analysis(analyzer, 'axis1') for comprehensive view")
#     print("Use: visualize_specific_profile_comparison(analyzer, 'axis1', profile_index) for detailed comparison")
#     print("Use: quick_reference_overview(analyzer, 'axis1') for text summary")

"""
Analyze the neighbors-based info

"""

import numpy as np
from collections import defaultdict

def calculate_neighbor_based_differences(nei1_S, link_matrix_1, comparisons, 
                                       use_significant_links=True):
    """
    Calculate neighbor-based average differences for peak comparison metrics.
    
    Parameters:
    -----------
    nei1_S : list of arrays
        List of length N, each array contains the neighbors of each point
    link_matrix_1 : numpy.ndarray
        Link matrix with shape (N, N), where 1 indicates significant relationship
    comparisons : list of dict
        Peak comparison results from analyzer.compare_with_reference()
        Each dict contains 'profile_index' and 'shape_descriptors'
    use_significant_links : bool, default=True
        If True, only consider neighbors where link_matrix_1[i,j] == 1
        If False, use all neighbors in nei1_S regardless of link_matrix_1
    
    Returns:
    --------
    dict : Dictionary with neighbor-based averages for each comparison metric
    """
    
    N = len(nei1_S)
    
    # Initialize result dictionary to store neighbor-based averages
    neighbor_averages = {}
    
    # Extract all possible metric paths from the comparisons structure
    metric_paths = _extract_metric_paths(comparisons)
    
    # Initialize arrays for each metric
    for path in metric_paths:
        neighbor_averages[path] = np.full(N, np.nan)
    
    # Process each point
    for point_idx in range(N):
        # Get neighbors for this point
        neighbors = nei1_S[point_idx]
        
        # Filter neighbors based on use_significant_links setting
        if use_significant_links:
            # Only use neighbors with significant relationships (link_matrix_1 == 1)
            valid_neighbors = []
            for neighbor_idx in neighbors:
                if neighbor_idx < N and link_matrix_1[point_idx, neighbor_idx] == 1:
                    valid_neighbors.append(neighbor_idx)
        else:
            # Use all neighbors regardless of link_matrix_1
            valid_neighbors = [idx for idx in neighbors if idx < N]
        
        if len(valid_neighbors) == 0:
            # No valid neighbors, skip this point (values remain NaN)
            continue
        
        # Get current point's metrics
        current_metrics = _extract_metrics_from_comparison(comparisons[point_idx])
        
        # Collect neighbor metrics
        neighbor_metrics_list = []
        for neighbor_idx in valid_neighbors:
            neighbor_metrics = _extract_metrics_from_comparison(comparisons[neighbor_idx])
            neighbor_metrics_list.append(neighbor_metrics)
        
        # Calculate absolute differences and averages
        for path in metric_paths:
            current_value = current_metrics.get(path, np.nan)
            
            if np.isnan(current_value):
                continue
            
            neighbor_differences = []
            for neighbor_metrics in neighbor_metrics_list:
                neighbor_value = neighbor_metrics.get(path, np.nan)
                if not np.isnan(neighbor_value):
                    # Simple absolute difference: current - neighbor
                    diff_value = current_value - neighbor_value
                    neighbor_differences.append(diff_value)
            
            # Average the differences with neighbors
            if len(neighbor_differences) > 0:
                neighbor_averages[path][point_idx] = np.nanmean(neighbor_differences)
    
    return neighbor_averages


def _extract_metric_paths(comparisons):
    """
    Extract all possible metric paths from the comparison structure.
    
    Returns:
    --------
    list : List of string paths to all metrics
    """
    if not comparisons or len(comparisons) == 0:
        return []
    
    # Use the first valid comparison to determine structure
    sample_comparison = None
    for comp in comparisons:
        if 'shape_descriptors' in comp and comp['shape_descriptors']:
            sample_comparison = comp['shape_descriptors']
            break
    
    if sample_comparison is None:
        return []
    
    paths = []
    
    # Extract paths recursively
    def extract_paths_recursive(data, current_path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if isinstance(value, dict):
                    extract_paths_recursive(value, new_path)
                else:
                    # Leaf node - this is a metric
                    paths.append(new_path)
        elif isinstance(data, (list, tuple)):
            # Handle lists/tuples if present
            for i, item in enumerate(data):
                new_path = f"{current_path}[{i}]"
                extract_paths_recursive(item, new_path)
    
    extract_paths_recursive(sample_comparison)
    return paths


def _extract_metrics_from_comparison(comparison_result):
    """
    Extract all metrics from a single comparison result.
    
    Parameters:
    -----------
    comparison_result : dict
        Single comparison result with 'shape_descriptors'
    
    Returns:
    --------
    dict : Flattened metrics with path keys
    """
    if 'shape_descriptors' not in comparison_result:
        return {}
    
    shape_descriptors = comparison_result['shape_descriptors']
    metrics = {}
    
    def flatten_dict(data, parent_path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{parent_path}.{key}" if parent_path else key
                if isinstance(value, dict):
                    flatten_dict(value, current_path)
                else:
                    metrics[current_path] = value
        elif isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                current_path = f"{parent_path}[{i}]"
                flatten_dict(item, current_path)
    
    flatten_dict(shape_descriptors)
    return metrics


def get_summary_statistics(neighbor_averages):
    """
    Calculate summary statistics for the neighbor-based differences.
    
    Parameters:
    -----------
    neighbor_averages : dict
        Result from calculate_neighbor_based_differences()
    
    Returns:
    --------
    dict : Summary statistics for each metric
    """
    summary = {}
    
    for metric_name, values in neighbor_averages.items():
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            summary[metric_name] = {
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'median': np.median(valid_values),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'count_valid': len(valid_values),
                'count_total': len(values)
            }
        else:
            summary[metric_name] = {
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count_valid': 0,
                'count_total': len(values)
            }
    
    return summary


# Example usage function
def analyze_neighbor_differences(nei1_S, link_matrix_1, comparisons, 
                               metrics_of_interest=None, use_significant_links=True, 
                               verbose=True):
    """
    Complete workflow for neighbor-based difference analysis.
    
    Parameters:
    -----------
    nei1_S : list of arrays
        Neighbor lists for each point
    link_matrix_1 : numpy.ndarray
        Link matrix indicating significant relationships
    comparisons : list of dict
        Peak comparison results
    metrics_of_interest : list, optional
        Specific metrics to focus on. If None, analyzes all metrics.
    use_significant_links : bool, default=True
        If True, only consider neighbors where link_matrix_1[i,j] == 1
        If False, use all neighbors in nei1_S regardless of link_matrix_1
    verbose : bool
        Whether to print summary information
    
    Returns:
    --------
    dict : Complete analysis results
    """
    
    # Calculate neighbor-based differences
    neighbor_averages = calculate_neighbor_based_differences(
        nei1_S, link_matrix_1, comparisons, use_significant_links=use_significant_links
    )
    
    # Filter metrics if specified
    if metrics_of_interest:
        filtered_averages = {}
        for metric in metrics_of_interest:
            if metric in neighbor_averages:
                filtered_averages[metric] = neighbor_averages[metric]
            else:
                if verbose:
                    print(f"Warning: Metric '{metric}' not found in results")
        neighbor_averages = filtered_averages
    
    # Calculate summary statistics
    summary_stats = get_summary_statistics(neighbor_averages)
    
    if verbose:
        neighbor_type = "significant neighbors" if use_significant_links else "all neighbors"
        print(f"=== Neighbor-based Difference Analysis ===")
        print(f"Total points: {len(nei1_S)}")
        print(f"Using: {neighbor_type}")
        print(f"Metrics analyzed: {len(neighbor_averages)}")
        if use_significant_links:
            print(f"Total significant links: {np.sum(link_matrix_1)}")
        print()
        
        # Show summary for key metrics
        key_metrics = [
            'similarity_measures.correlation',
            'similarity_measures.cosine_similarity', 
            'moment_differences.width_ratio',
            'moment_differences.centroid_diff',
            'auc_comparison.auc_ratio'
        ]
        
        for metric in key_metrics:
            if metric in summary_stats:
                stats = summary_stats[metric]
                print(f"{metric}:")
                print(f"  Mean relative difference: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Valid points: {stats['count_valid']}/{stats['count_total']}")
                print()
    
    return {
        'neighbor_averages': neighbor_averages,
        'summary_statistics': summary_stats,
        'input_info': {
            'n_points': len(nei1_S),
            'n_metrics': len(neighbor_averages),
            'use_significant_links': use_significant_links,
            'significant_links': np.sum(link_matrix_1) if use_significant_links else 'N/A'
        }
    }

def calculate_current(comparisons, S):
    """
    Calculate neighbor-based average differences for peak comparison metrics.
    
    Parameters:
    -----------
    comparisons : list of dict
        Peak comparison results from analyzer.compare_with_reference()
        Each dict contains 'profile_index' and 'shape_descriptors'
    S : array, the coordinate info
    
    Returns:
    --------
    dict : Dictionary with neighbor-based averages for each comparison metric
    """
    
    N = len(S)
    
    # Initialize result dictionary to store values of current point
    current_values = {}
    
    # Extract all possible metric paths from the comparisons structure
    metric_paths = _extract_metric_paths(comparisons)
    
    # Initialize arrays for each metric
    for path in metric_paths:
        current_values[path] = np.full(N, np.nan)
    
    # Process each point
    for point_idx in range(N):
        # Get current point's metrics
        current_metrics = _extract_metrics_from_comparison(comparisons[point_idx])
 
        # Calculate absolute differences and averages
        for path in metric_paths:
            current_value = current_metrics.get(path, np.nan)
            
            if np.isnan(current_value):
                continue

            current_values[path][point_idx] = current_value
    
    return current_values

def analyze_current_value(comparisons, S, metrics_of_interest = None, verbose = True):
    """
    Complete workflow for point-based analysis.
    
    Parameters:
    -----------
    comparisons : list of dict
        Peak comparison results
    S : array, the coordinate info (which is not really necessary)
    
    verbose : bool
        Whether to print summary information

    Returns:
    --------
    dict : Complete analysis results
    """
    
    # Calculate neighbor-based differences
    current_values = calculate_current(
        comparisons, S=S
    )
    
    # Filter metrics if specified
    if metrics_of_interest:
        filtered_values = {}
        for metric in metrics_of_interest:
            if metric in current_values:
                filtered_values[metric] = current_values[metric]
            else:
                if verbose:
                    print(f"Warning: Metric '{metric}' not found in results")
        current_values = filtered_values
    
    # Calculate summary statistics
    summary_stats = get_summary_statistics(current_values)
    
    if verbose:
        data_type = "Point-dependent"
        print(f"=== Neighbor-based Difference Analysis ===")
        print(f"Total points: {len(S)}")
        print(f"Using: {data_type}")
        print(f"Metrics analyzed: {len(current_values)}")

        print()
        
        # Show summary for key metrics
        key_metrics = [
            'similarity_measures.correlation',
            'similarity_measures.cosine_similarity', 
            'moment_differences.width_ratio',
            'moment_differences.centroid_diff',
            'auc_comparison.auc_ratio'
        ]
        
        for metric in key_metrics:
            if metric in summary_stats:
                stats = summary_stats[metric]
                print(f"{metric}:")
                print(f"  Mean relative difference: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Valid points: {stats['count_valid']}/{stats['count_total']}")
                print()
    
    return {
        'point_based': current_values,
        'summary_statistics': summary_stats,
        'input_info': {
            'n_points': len(S),
            'n_metrics': len(current_values),
        }
    }
# Example of how to use with your workflow:
# """
# # Method 1: Using only significant neighbors (default)
# neighbor_results_significant = analyze_neighbor_differences(
#     nei1_S=nei1_S,
#     link_matrix_1=link_matrix_1, 
#     comparisons=comparisons,
#     use_significant_links=True,  # Default behavior
#     verbose=True
# )

# # Method 2: Using all neighbors regardless of significance
# neighbor_results_all = analyze_neighbor_differences(
#     nei1_S=nei1_S,
#     link_matrix_1=link_matrix_1, 
#     comparisons=comparisons,
#     use_significant_links=False,  # Use all neighbors
#     verbose=True
# )

# # Access specific results:
# correlation_differences = neighbor_results_significant['neighbor_averages']['similarity_measures.correlation']
# width_ratio_differences = neighbor_results_all['neighbor_averages']['moment_differences.width_ratio']

# # Get summary statistics:
# summary = neighbor_results_significant['summary_statistics']
# """

# visualization of variation
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_regional_metrics(ax, data_metric_dict, region_labels, selected_regions, selected_metrics, colors):
    """
    Plot scatter plot showing how metrics vary across regions (mean values).
    Each metric gets its own Y-axis using twinx().
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The base axes to plot on
    data_metric_dict : dict
        Dictionary where keys are metric names, values are arrays of shape (N,)
    region_labels : array-like
        Array of region labels for each data point, shape (N,)
    selected_regions : array-like
        Array of region labels to include in the plot
    selected_metrics : list
        List of metric names to include in the plot
    """
    
    # Calculate mean values for each region and metric
    region_means = {}
    for region in selected_regions:
        region_mask = np.array(region_labels) == region
        region_means[region] = {}
        for metric in selected_metrics:
            if metric in data_metric_dict:
                metric_data = np.array(data_metric_dict[metric])
                region_means[region][metric] = np.mean(metric_data[region_mask])
    
    # Create axes for each metric
    axes = [ax]  # First metric uses the original axis
    for i in range(1, len(selected_metrics)):
        axes.append(ax.twinx())
    
    # Position additional Y-axes to avoid overlap
    for i, axis in enumerate(axes[2:], start=2):
        axis.spines['right'].set_position(('outward', 67 * (i - 1)))
    
    # Colors for regions
    region_colors = colors
    
    # Plot each metric on its own Y-axis
    for metric_idx, metric in enumerate(selected_metrics):
        current_ax = axes[metric_idx]
        x_pos = metric_idx
        
        # Plot each region for this metric
        for region_idx, region in enumerate(selected_regions):
            y_value = region_means[region][metric]
            current_ax.scatter(x_pos, y_value, 
                             color=region_colors[region_idx],
                             label=f'Region {region}' if metric_idx == 0 else "",
                             s=60, alpha=0.8)
        
        # Customize Y-axis for this metric
        current_ax.set_ylabel(f'{metric}', color='black')
        current_ax.tick_params(axis='y', labelcolor='black')
        

        # Set Y-axis limits with some padding
        metric_values = [region_means[region][metric] for region in selected_regions]
        y_min, y_max = min(metric_values), max(metric_values)
        y_range = y_max - y_min
        padding = 0.1 * y_range if y_range > 0 else 0.1 * abs(y_max)
        current_ax.set_ylim(y_min - padding, y_max + padding)
    
    # Customize the main axis (X-axis)
    ax.set_xticks(np.arange(len(selected_metrics)))
    ax.set_xticklabels(selected_metrics,rotation=45, ha='right')
    ax.set_xlabel('Metrics')
    
    # Only show legend on the first axis to avoid duplication
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, -0.05))
    ax.grid(True, alpha=0.3)
    ax.set_title('Regional Metrics Comparison (Independent Y-axes)')
    
    return axes
