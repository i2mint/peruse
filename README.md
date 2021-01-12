
# peruse
Explore a waveform with slang


To install:	```pip install peruse```


# Example

```python
from numpy import *
from hum import disp_wf, plot_wf
import soundfile as sf
from peruse.single_wf_snip_analysis import TaggedWaveformAnalysisExtended
import os
import pickle

try:
    from slang.util_data import displayable_unichr
    unichr_code_of_snip = array(displayable_unichr
                                + list(unique(list(set(range(33, 20000)).difference(displayable_unichr)))))
    snip_of_unichr_code = (nan * empty(unichr_code_of_snip.max() + 1)).astype(int)
    snip_of_unichr_code[unichr_code_of_snip] = arange(len(unichr_code_of_snip))

    def snip_to_str(snip):
        return chr(unichr_code_of_snip[snip])
except ImportError as e:
    def snip_to_str(snip):
        return chr(33 + snip)
    
def string_of_snips(snips):
    return "".join(map(snip_to_str, snips))
```

```python
filepath = "Enter audio filepath here"
wf, sr = sf.read(filepath)
disp_wf(wf, sr)
```

## Unsupervised

Perhaps you just want to get a perspective on your sound, without specifying annotations.

Perhaps you don't know what to annotate and you want snips to help you find patterns to annotate.

Fit the snipper

```python
from peruse.single_wf_snip_analysis import TaggedWaveformAnalysisExtended

tw = TaggedWaveformAnalysisExtended(sr=sr, 
                                    tile_size_frm=2048, 
                                    chk_size_frm=43008, 
                                    prior_count=1)
tw.fit(wf)
```

Get the snips of a waveform (here the same as fit with, but could be another)

```python
snips = tw.snips_of_wf(wf)
len(snips), len(unique(snips))
```

View them as characters

```python
print(string_of_snips(snips))
```

```
annkncckkobbjhihjacjcjeneijaeofgibiecikjjkdnhajekachbobchoadkjjjjknkkkjofglmijaiinieajdccnkjnollil
hjhkokejkeacdkcofgaldikljnekdijekkjieoboocinchjadannadnnjofgiiielaekheiccnkkeejlbbllichckkinojaeoa
aofgllllnadkdkcneian
```

Plot (inverse of) snip probabilities (says how rare they are (outliers) from the perspective of the wf that was used to fit, and gives SOME view of the sound)

```python
tw.plot_tiles(1 / array(list(map(tw.prob_of_snip.get, snips))));
```

```python
tw.plot_tiles(log(1 / array(list(map(tw.prob_of_snip.get, snips)))));
```

