# Artificial Cortex
### Max Kleinsasser

The Artificial Cortex is the name I've given to a learnable data structure and algorithm of my creation, which was heavily inspired by the Thousand Brains Theory of Intelligence. I understand "cortex" is not a very specific or applicable word on its own (I took it from "cerebral cortex"), but the AI community already has an established definition for the term "neural network" despite it describing my thing perfectly. The rest of this document will be more concise.

The Artificial Cortex is designed to address three of the biggest problems in state-of-the-art AI methods, Deep Learning in particular. Specifically, SOTA AI models:
1. Require a significant amount of labeled training data.
2. Accomplish only narrowly-defined tasks, unable to generalize from acquired "knowledge".
3. Must be treated as black-boxes, the reasons for their outputs largely unknowable.

I believe The Artificial Cortex makes progress in all of these areas with varying stride-lengths.

## AC Requires Less Data

A percursor to "does it learn fast?" is, of course, "does it learn?". The AC alone is capable of achieving underwhelming-to-pretty-good accuracy on various classification tasks without restricting the training data, though it was design to be augmented with pre-processing:

Iris Dataset: 97%
MNIST Dataset (Raw Pixels): 95%
Fashion MNIST Dataset (Raw Pixels): 82%

# cortex
