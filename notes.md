# Notes and TODOs

*These are the haphaserds notes we take during meetings are studies.*

* We should compare our "results", of any kind, with the random agent
so that we have a benchmark.

The following should be use to generate the environment

```python
env = gym.make('RevealMNIST-v0',
               classifier_model_weights_loc="mnist_predictor_masked.pt",
               device='cpu',
               visualize=True)
```

*Note that the device can change*

## from the source code
the agent can fail up to 3 times during an eposide\
...

## TODO

* **One should try to code as modular as possible
so that others can chime in by implementing unfinished methods**

* We decided on implementing a DQN approach.
We will work on it separately and the merge the work.
