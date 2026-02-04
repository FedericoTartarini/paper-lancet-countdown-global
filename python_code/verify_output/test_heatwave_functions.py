"""
Test script for the heatwave calculation functions in d_calculate_heatwaves.py.

This script creates sample temperature data and tests the vectorized heatwave
calculation functions to ensure they work correctly.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from python_code.weather.d_calculate_heatwaves import (
    count_heatwave_days_vectorized,
    calculate_heatwave_metrics_vectorized,
)


def create_sample_data():
    """Create sample temperature data for testing."""
    # Create a simple 10-day, 2x2 grid dataset
    time = xr.date_range("2020-01-01", periods=10, freq="D")
    lat = [0, 1]
    lon = [0, 1]

    # Sample temperature data (in Kelvin)
    # Create a pattern where days 3-7 are hot
    t_max_data = np.full((10, 2, 2), 300)  # Base 27°C
    t_min_data = np.full((10, 2, 2), 295)  # Base 22°C

    # Make days 3-7 hot (above thresholds)
    t_max_data[3:8, :, :] = 310  # 37°C
    t_min_data[3:8, :, :] = 305  # 32°C

    # Create xarray DataArrays
    t_max = xr.DataArray(
        t_max_data,
        coords={"time": time, "latitude": lat, "longitude": lon},
        name="t_max",
    )
    t_min = xr.DataArray(
        t_min_data,
        coords={"time": time, "latitude": lat, "longitude": lon},
        name="t_min",
    )

    # Thresholds (95th percentile values)
    t_max_threshold = xr.DataArray(
        [[305, 305], [305, 305]],  # 32°C threshold
        coords={"latitude": lat, "longitude": lon},
        name="t_max_threshold",
    )
    t_min_threshold = xr.DataArray(
        [[300, 300], [300, 300]],  # 27°C threshold
        coords={"latitude": lat, "longitude": lon},
        name="t_min_threshold",
    )

    return t_max, t_min, t_max_threshold, t_min_threshold


def test_heatwave_days():
    """Test the count_heatwave_days_vectorized function."""
    print("🧪 Testing count_heatwave_days_vectorized...")

    t_max, t_min, t_max_thresh, t_min_thresh = create_sample_data()

    # Run the function
    hw_days = count_heatwave_days_vectorized(
        t_max, t_min, t_max_thresh, t_min_thresh, hw_min_length=3
    )

    # Expected: Days 3-7 should be heatwave days (5 days total)
    # Days 0-2: Not hot enough
    # Days 3-7: Hot (5 consecutive days >= min length)
    # Days 8-9: Not hot enough

    expected_hw_days = np.zeros((10, 2, 2), dtype=bool)
    expected_hw_days[3:8, :, :] = True  # Days 3-7

    # Check if results match expectations
    matches = np.array_equal(hw_days.values, expected_hw_days)

    if matches:
        print("   ✅ Heatwave days calculation correct")
        print(f"   📊 Total heatwave days: {hw_days.sum().values}")
    else:
        print("   ❌ Heatwave days calculation incorrect")
        print(f"   Expected shape: {expected_hw_days.shape}")
        print(f"   Got shape: {hw_days.values.shape}")
        return False

    return True


def test_heatwave_metrics():
    """Test the calculate_heatwave_metrics_vectorized function."""
    print("🧪 Testing calculate_heatwave_metrics_vectorized...")

    t_max, t_min, t_max_thresh, t_min_thresh = create_sample_data()

    # Run the function
    results = calculate_heatwave_metrics_vectorized(
        t_max, t_min, t_max_thresh, t_min_thresh, hw_min_length=3
    )

    # Check output structure
    required_vars = {"heatwave_count", "heatwave_days"}
    if not required_vars.issubset(set(results.data_vars)):
        print(
            f"   ❌ Missing required variables: {required_vars - set(results.data_vars)}"
        )
        return False

    # Check dimensions
    expected_dims = {"latitude", "longitude"}
    if not expected_dims.issubset(set(results.dims)):
        print(f"   ❌ Missing dimensions: {expected_dims - set(results.dims)}")
        return False

    # Check values
    hw_count = results["heatwave_count"]
    hw_days = results["heatwave_days"]

    # Expected: 1 heatwave event, 5 heatwave days per location
    expected_count = 1
    expected_days = 5

    if (
        hw_count.values[0, 0] == expected_count
        and hw_days.values[0, 0] == expected_days
    ):
        print("   ✅ Heatwave metrics calculation correct")
        print(f"   📊 Heatwave count: {hw_count.values[0, 0]}")
        print(f"   📊 Heatwave days: {hw_days.values[0, 0]}")
        return True
    else:
        print("   ❌ Heatwave metrics calculation incorrect")
        print(f"   Expected count: {expected_count}, got: {hw_count.values[0, 0]}")
        print(f"   Expected days: {expected_days}, got: {hw_days.values[0, 0]}")
        return False


def test_edge_cases():
    """Test edge cases."""
    print("🧪 Testing edge cases...")

    # Test with no heatwaves
    time = xr.date_range("2020-01-01", periods=10, freq="D")
    lat = [0, 1]
    lon = [0, 1]

    # All temperatures below threshold
    t_max = xr.DataArray(
        np.full((10, 2, 2), 295),  # 22°C
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    t_min = xr.DataArray(
        np.full((10, 2, 2), 290),  # 17°C
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    t_max_thresh = xr.DataArray(
        [[300, 300], [300, 300]], coords={"latitude": lat, "longitude": lon}
    )
    t_min_thresh = xr.DataArray(
        [[295, 295], [295, 295]], coords={"latitude": lat, "longitude": lon}
    )

    results = calculate_heatwave_metrics_vectorized(
        t_max, t_min, t_max_thresh, t_min_thresh
    )

    if results["heatwave_count"].sum() == 0 and results["heatwave_days"].sum() == 0:
        print("   ✅ No heatwaves case handled correctly")
        return True
    else:
        print("   ❌ No heatwaves case failed")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("🔬 Heatwave Calculation Function Tests")
    print("=" * 70)

    tests = [test_heatwave_days, test_heatwave_metrics, test_edge_cases]
    passed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            print()

    print("=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Heatwave functions are working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")

    print("=" * 70)


if __name__ == "__main__":
    main()
