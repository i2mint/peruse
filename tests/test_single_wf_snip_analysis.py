"""Tests for single_wf_snip_analysis module"""

import pytest
import numpy as np
from numpy import sin, pi, concatenate, linspace
from peruse.single_wf_snip_analysis import (
    TaggedWaveformAnalysis,
    abs_stft,
    spectr,
    log_spectr,
    annots_type,
    segment_tag_item_type,
    tag_segments_dict_from_tag_segment_list,
    segment_tag_it_from_tag_segments_dict,
    count_to_prob,
    running_mean,
    ChkUnitConverter,
)

# Try to import extended class (may not be available without hum)
try:
    from peruse.single_wf_snip_analysis import TaggedWaveformAnalysisExtended
    HAS_EXTENDED = True
except ImportError:
    HAS_EXTENDED = False


# ================================================================================================
# Test Fixtures - Generate synthetic waveforms
# ================================================================================================

@pytest.fixture
def simple_sine_wf():
    """Generate a simple sine wave for testing"""
    sr = 44100
    duration = 1.0  # 1 second
    freq = 440  # A4 note
    t = linspace(0, duration, int(sr * duration))
    wf = sin(2 * pi * freq * t)
    return wf, sr


@pytest.fixture
def multi_freq_wf():
    """Generate a waveform with multiple frequency components"""
    sr = 44100
    duration = 0.5
    t = linspace(0, duration, int(sr * duration))
    # Combine multiple frequencies
    wf = (sin(2 * pi * 220 * t) +
          0.5 * sin(2 * pi * 440 * t) +
          0.3 * sin(2 * pi * 880 * t))
    return wf, sr


@pytest.fixture
def tagged_waveforms():
    """Generate waveforms with tags for supervised learning"""
    sr = 44100
    duration = 0.2

    # Create three different "sound types"
    t = linspace(0, duration, int(sr * duration))
    wf_low = sin(2 * pi * 220 * t)  # Low frequency
    wf_mid = sin(2 * pi * 440 * t)  # Mid frequency
    wf_high = sin(2 * pi * 880 * t)  # High frequency

    # Concatenate them
    full_wf = concatenate([wf_low, wf_mid, wf_high])

    # Create tag annotations
    tag_segments = {
        'low': [(0.0, 0.2)],
        'mid': [(0.2, 0.4)],
        'high': [(0.4, 0.6)]
    }

    return full_wf, sr, tag_segments


# ================================================================================================
# Tests for ChkUnitConverter
# ================================================================================================

class TestChkUnitConverter:
    """Tests for time unit conversion"""

    def test_initialization(self):
        """Test converter initialization with default parameters"""
        converter = ChkUnitConverter()
        assert converter.sr == 44100
        assert converter.buf_size_frm == 2048
        assert converter.chk_size_frm == 2048 * 21

    def test_custom_initialization(self):
        """Test converter with custom parameters"""
        converter = ChkUnitConverter(sr=22050, buf_size_frm=1024, chk_size_frm=1024 * 10)
        assert converter.sr == 22050
        assert converter.buf_size_frm == 1024
        assert converter.chk_size_frm == 1024 * 10

    def test_frames_to_buffers(self):
        """Test conversion from frames to buffers"""
        converter = ChkUnitConverter(buf_size_frm=1024)
        result = converter(2048, 'frm', 'buf')
        assert result == 2  # 2048 frames = 2 buffers of 1024

    def test_seconds_to_frames(self):
        """Test conversion from seconds to frames"""
        converter = ChkUnitConverter(sr=44100)
        result = converter(1.0, 's', 'frm')
        assert result == 44100  # 1 second = 44100 frames at 44.1kHz

    def test_invalid_conversion_raises_error(self):
        """Test that invalid unit conversion raises ValueError"""
        converter = ChkUnitConverter()
        with pytest.raises(ValueError, match="didn't have a way to convert"):
            converter(100, 'invalid_unit', 'frm')


# ================================================================================================
# Tests for spectral analysis functions
# ================================================================================================

class TestSpectralFunctions:
    """Tests for STFT and spectrogram functions"""

    def test_abs_stft_basic(self, simple_sine_wf):
        """Test basic STFT computation"""
        wf, _ = simple_sine_wf
        result = abs_stft(wf[:4096], tile_size=2048)
        assert result.shape[0] == 1025  # (n_fft // 2) + 1
        assert result.shape[1] >= 1

    def test_abs_stft_empty_waveform(self):
        """Test STFT with empty waveform"""
        result = abs_stft(np.array([]))
        assert len(result) == 0

    def test_spectr_shape(self, simple_sine_wf):
        """Test spectrogram computation returns correct shape"""
        wf, sr = simple_sine_wf
        result = spectr(wf[:4096], sr=sr, tile_size=2048)
        assert result.ndim == 2
        assert result.shape[1] == 1025  # Frequency bins

    def test_log_spectr_no_inf(self, simple_sine_wf):
        """Test log spectrogram doesn't produce inf values"""
        wf, sr = simple_sine_wf
        result = log_spectr(wf[:4096], sr=sr, tile_size=2048)
        assert not np.any(np.isinf(result))
        assert not np.any(np.isnan(result))


# ================================================================================================
# Tests for annotation handling
# ================================================================================================

class TestAnnotationFunctions:
    """Tests for annotation type detection and conversion"""

    def test_annots_type_tag_segments(self):
        """Test detection of tag_segments dictionary format"""
        annots = {'tag1': [(0, 1), (2, 3)], 'tag2': [(1, 2)]}
        result = annots_type(annots)
        assert result == 'tag_segments'

    def test_segment_tag_item_type_bt_tt_tag_tuple(self):
        """Test detection of (bt, tt, tag) tuple format"""
        item = (0.0, 1.0, 'tag1')
        result = segment_tag_item_type(item)
        assert result == 'bt_tt_tag_tuple'

    def test_segment_tag_item_type_tag_bt_tt_tuple(self):
        """Test detection of (tag, bt, tt) tuple format"""
        item = ('tag1', 0.0, 1.0)
        result = segment_tag_item_type(item)
        assert result == 'tag_bt_tt_tuple'

    def test_segment_tag_item_type_segment_tag_tuple(self):
        """Test detection of ((bt, tt), tag) tuple format"""
        item = ((0.0, 1.0), 'tag1')
        result = segment_tag_item_type(item)
        assert result == 'segment_tag_tuple'

    def test_tag_segments_dict_conversion(self):
        """Test conversion from list to dict format"""
        tag_segment_list = [((0.0, 1.0), 'tag1'), ((1.0, 2.0), 'tag2'), ((2.0, 3.0), 'tag1')]
        result = tag_segments_dict_from_tag_segment_list(tag_segment_list)
        assert 'tag1' in result
        assert 'tag2' in result
        assert len(result['tag1']) == 2
        assert len(result['tag2']) == 1

    def test_segment_tag_it_from_dict(self):
        """Test iterator creation from tag_segments dict"""
        tag_segments = {'tag1': [(0.0, 1.0), (2.0, 3.0)], 'tag2': [(1.0, 2.0)]}
        result = list(segment_tag_it_from_tag_segments_dict(tag_segments))
        assert len(result) == 3
        assert all(len(item) == 2 for item in result)  # Each item is (segment, tag)


# ================================================================================================
# Tests for utility functions
# ================================================================================================

class TestUtilityFunctions:
    """Tests for helper functions"""

    def test_count_to_prob_basic(self):
        """Test probability calculation from counts"""
        count_of_item = {0: 10, 1: 20, 2: 30}
        item_set = [0, 1, 2]
        result = count_to_prob(count_of_item, item_set, prior_count=1)

        # Check probabilities sum to 1
        assert abs(sum(result.values()) - 1.0) < 1e-10
        # Check probabilities are in correct order (with prior)
        assert result[2] > result[1] > result[0]

    def test_count_to_prob_with_missing_items(self):
        """Test probability calculation when some items have no counts"""
        count_of_item = {0: 10}
        item_set = [0, 1, 2]
        result = count_to_prob(count_of_item, item_set, prior_count=1)

        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert result[0] > result[1]  # Item 0 has more count
        assert result[1] == result[2]  # Items 1 and 2 only have prior count

    def test_running_mean_basic(self):
        """Test running mean calculation"""
        data = [1, 3, 5, 7, 9]
        result = list(running_mean(data, chk_size=2))
        expected = [2.0, 4.0, 6.0, 8.0]
        assert result == expected

    def test_running_mean_with_step(self):
        """Test running mean with step > 1"""
        data = [1, 3, 5, 7, 9]
        result = list(running_mean(data, chk_size=2, chk_step=2))
        expected = [2.0, 6.0]
        assert result == expected

    def test_running_mean_size_one(self):
        """Test running mean with window size 1 (identity)"""
        data = [1, 2, 3, 4]
        result = list(running_mean(data, chk_size=1))
        assert result == data


# ================================================================================================
# Tests for TaggedWaveformAnalysis
# ================================================================================================

class TestTaggedWaveformAnalysis:
    """Tests for the main TaggedWaveformAnalysis class"""

    def test_initialization_defaults(self):
        """Test initialization with default parameters"""
        twa = TaggedWaveformAnalysis()
        assert twa.sr == 44100
        assert twa.tile_size_frm == 2048
        assert twa.chk_size_frm == 2048 * 21
        assert twa.prior_count == 1
        assert twa.n_snips is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters"""
        twa = TaggedWaveformAnalysis(sr=22050, tile_size_frm=1024, n_snips=50)
        assert twa.sr == 22050
        assert twa.tile_size_frm == 1024
        assert twa.n_snips == 50

    def test_fit_unsupervised(self, simple_sine_wf):
        """Test fitting with unsupervised data (no annotations)"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr)
        twa.fit(wf)

        # Check that model was fitted
        assert twa.fvs_to_snips is not None
        assert twa.snips is not None
        assert len(twa.snips) > 0
        assert twa.n_snips is not None
        assert twa.n_snips > 0

    def test_fit_supervised(self, tagged_waveforms):
        """Test fitting with supervised data (with annotations)"""
        wf, sr, tag_segments = tagged_waveforms
        # Use fewer components for LDA since we only have 3 classes
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        twa = TaggedWaveformAnalysis(sr=sr, fv_tiles_model=LDA(n_components=2))
        twa.fit(wf, annots_for_tag=tag_segments)

        # Check that model was fitted and learned tags
        assert twa.fvs_to_snips is not None
        assert twa.tag_count_for_snip is not None
        assert len(twa.tag_count_for_snip) > 0
        assert hasattr(twa, 'classes_')
        assert len(twa.classes_) == 3  # We have 3 tags

    def test_snips_of_wf(self, simple_sine_wf):
        """Test snip extraction from waveform"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr, n_snips=10)
        twa.fit(wf[:sr // 2])  # Fit on half

        # Extract snips from the waveform
        snips = twa.snips_of_wf(wf[:sr // 2])
        assert len(snips) > 0
        assert all(0 <= s < 10 for s in snips)  # Snips should be in range [0, n_snips)

    def test_tiles_of_wf(self, simple_sine_wf):
        """Test tile extraction from waveform"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr, tile_size_frm=2048)
        tiles = twa.tiles_of_wf(wf[:sr // 2])

        assert tiles.ndim == 2
        assert tiles.shape[1] == 1025  # (tile_size // 2) + 1

    def test_prob_of_snip(self, simple_sine_wf):
        """Test snip probability calculation"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr, n_snips=10)
        twa.fit(wf)

        prob_dict = twa.prob_of_snip
        # Check probabilities sum to approximately 1
        assert abs(sum(prob_dict.values()) - 1.0) < 0.01
        # Check all probabilities are positive
        assert all(p > 0 for p in prob_dict.values())

    def test_count_of_snip(self, simple_sine_wf):
        """Test snip count calculation"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr, n_snips=10)
        twa.fit(wf)

        count_dict = twa.count_of_snip
        assert len(count_dict) > 0
        assert all(isinstance(v, (int, np.integer)) for v in count_dict.values())
        assert all(v > 0 for v in count_dict.values())

    def test_get_wf_for_bt_tt(self, simple_sine_wf):
        """Test waveform segment extraction by time"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr)

        # Extract 0.1 second segment
        segment = twa.get_wf_for_bt_tt(wf, bt=0.0, tt=0.1)
        expected_length = int(0.1 * sr)
        assert len(segment) == expected_length

    def test_fit_with_explicit_n_snips(self, simple_sine_wf):
        """Test fitting with explicitly specified number of snips"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr)
        twa.fit(wf, n_snips=15)

        assert twa.n_snips == 15
        assert twa.fvs_to_snips.n_clusters == 15

    def test_conditional_prob_ratio_uniform(self, simple_sine_wf):
        """Test conditional probability ratio with uniform prior"""
        wf, sr = simple_sine_wf
        twa = TaggedWaveformAnalysis(sr=sr, n_snips=10)
        twa.fit(wf)

        ratio = twa.conditional_prob_ratio_of_snip()
        assert len(ratio) > 0
        # Ratios should be positive
        assert all(v > 0 for v in ratio.values())

    def test_tag_probs_for_snips(self, tagged_waveforms):
        """Test tag probability retrieval for snips"""
        wf, sr, tag_segments = tagged_waveforms
        # Use fewer components for LDA since we only have 3 classes
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        twa = TaggedWaveformAnalysis(sr=sr, n_snips=10, fv_tiles_model=LDA(n_components=2))
        twa.fit(wf, annots_for_tag=tag_segments)

        snips = twa.snips_of_wf(wf)
        tag_probs = twa.tag_probs_for_snips(snips[:5])

        assert len(tag_probs) == 5
        # Each snip should have probability dict for all tags
        assert all(isinstance(p, dict) for p in tag_probs)


# ================================================================================================
# Tests for TaggedWaveformAnalysisExtended
# ================================================================================================

@pytest.mark.skipif(not HAS_EXTENDED, reason="TaggedWaveformAnalysisExtended requires hum package")
class TestTaggedWaveformAnalysisExtended:
    """Tests for the extended class with plotting capabilities"""

    def test_initialization(self):
        """Test that extended class initializes properly"""
        twa_ext = TaggedWaveformAnalysisExtended()
        assert twa_ext.sr == 44100
        # Extended class should have same attributes as base
        assert hasattr(twa_ext, 'tile_size_frm')
        assert hasattr(twa_ext, 'chk_size_frm')

    def test_plot_methods_exist(self):
        """Test that plotting methods exist"""
        twa_ext = TaggedWaveformAnalysisExtended()
        assert hasattr(twa_ext, 'plot_wf')
        assert hasattr(twa_ext, 'plot_tiles')
        assert hasattr(twa_ext, 'plot_tag_probs_for_snips')

    def test_fit_works_same_as_base(self, simple_sine_wf):
        """Test that extended class can fit data like base class"""
        wf, sr = simple_sine_wf
        twa_ext = TaggedWaveformAnalysisExtended(sr=sr)
        twa_ext.fit(wf)

        assert twa_ext.snips is not None
        assert len(twa_ext.snips) > 0


# ================================================================================================
# Integration tests
# ================================================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_unsupervised_workflow(self, multi_freq_wf):
        """Test complete workflow: fit, extract snips, get probabilities"""
        wf, sr = multi_freq_wf

        # Initialize and fit (use smaller n_snips to avoid KMeans error with small data)
        twa = TaggedWaveformAnalysis(sr=sr, n_snips=5)
        twa.fit(wf)

        # Extract snips from new data (same wf for this test)
        snips = twa.snips_of_wf(wf)

        # Get probabilities
        prob_dict = twa.prob_of_snip

        # Verify workflow produces valid results
        assert len(snips) > 0
        assert len(prob_dict) > 0
        assert all(s in prob_dict for s in set(snips))

    def test_complete_supervised_workflow(self, tagged_waveforms):
        """Test complete supervised workflow with tags"""
        wf, sr, tag_segments = tagged_waveforms

        # Initialize and fit with tags (use fewer components for LDA)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        twa = TaggedWaveformAnalysis(sr=sr, n_snips=15, fv_tiles_model=LDA(n_components=2))
        twa.fit(wf, annots_for_tag=tag_segments)

        # Extract snips
        snips = twa.snips_of_wf(wf)

        # Get tag probabilities
        tag_probs = twa.tag_probs_for_snips(snips[:10])

        # Verify results
        assert len(snips) > 0
        assert len(tag_probs) == 10
        assert hasattr(twa, 'classes_')
        assert len(twa.classes_) == 3  # Three tags: low, mid, high

    def test_fit_and_transform_different_waveforms(self, simple_sine_wf):
        """Test fitting on one waveform and transforming another"""
        wf1, sr = simple_sine_wf

        # Create second waveform with different frequency
        t = np.linspace(0, 0.5, int(sr * 0.5))
        wf2 = np.sin(2 * np.pi * 880 * t)

        # Fit on first waveform
        twa = TaggedWaveformAnalysis(sr=sr)
        twa.fit(wf1[:sr // 2])

        # Transform second waveform
        snips1 = twa.snips_of_wf(wf1[:sr // 4])
        snips2 = twa.snips_of_wf(wf2)

        # Both should produce valid snips
        assert len(snips1) > 0
        assert len(snips2) > 0
        assert all(0 <= s < twa.n_snips for s in snips1)
        assert all(0 <= s < twa.n_snips for s in snips2)
