# Character RNN Language Model in PyTorch

This is a character RNN-based language model in PyTorch. 
Code are based on examples in <https://github.com/pytorch/examples/tree/master/word_language_model>.
It can handle any Unicode corpus.

All configurations and hyper paramters are centerized in a JSON file (`hps/penn.json` is an example for PTB).
See the example for what are specified.

For training, use

```bash
./train.py --hps-file hps/penn.json
```

For sampling, use
```bash
./sample.py --hps-file hps/penn.json --nb-tokens 1000 --temperature 1.0
```
