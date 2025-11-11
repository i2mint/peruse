from peruse.single_wf_snip_analysis import TaggedWaveformAnalysis

try:
    from peruse.single_wf_snip_analysis import TaggedWaveformAnalysisExtended
except ImportError:
    # TaggedWaveformAnalysisExtended requires hum and matplotlib
    # If not available, only export base class
    pass