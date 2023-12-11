# Decoder-only transformers on Wikipedia

Simple project to get familiarity with open source tools. Training a
decoder-only transformer with classical sinus/cosinus positional encodings on
sequences of bytes randomly extracted from the
[enwik9 dataset](https://mattmahoney.net/dc/enwik9.zip).

No data or model parallelization implemented. The tensors and the model are
passed to the device of your choice, and you must configure it yourself.

## Installation and usage

```bash
git clone https://github.com/deepmind/language_modeling_is_compression.git
pip install -r requirements.txt
python3 main.py
```

Configs can be modified in `main.py` and are defined in `training.py` in the
`TrainConfig` dataclass.
