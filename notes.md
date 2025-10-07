* Take into account batch size vs epoch number subtle things e. g. lower loss at given epoch does not imply better HPO/Model
* Pass to LLM, get bullet point summary
* Estimate training costs for different config - e. g. maybe better to use more expensive model but for less hours 



Noor

* Loss vs epoch slope (to understand what to stop)
* Losses don't have the same range, hard to compare
* Loss spike don't know which datapoint responsible


Why muon optim so good?
gradient accum
GRPO

Activation checkpointing?

eval training vs train loss

wsd loss schedule


loss schedule takin into account when loss

loss peaks when and why? - should be deterministic based on data loader state


2 training comparison - check configs


pytorch profiler tensorboard


## Document AI

document viz + scores, filters ..