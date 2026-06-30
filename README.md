# E114 Paper

Preprint and figures for:

**A Single-Expert Readout of a Reflective Worldview Register in a Mixture-of-Experts Language Model**

Author: Jeffrey W. Shorthill  
Contact: jws299792@icloud.com

## Overview

This paper looks at a recurring shift in language model behavior: the moment a
model stops giving an ordinary task answer and starts writing from a reflective,
worldview-like stance. In that mode, the text is about meaning, belief, value,
existence, or what it is like for some target to have an inside.

The study uses Qwen3.5-35B-A3B, a mixture-of-experts model. These models route
each token through a small set of internal experts. We looked at whether one
specific expert, Expert 114, reliably rises when the model enters this reflective
style of writing.

To test that, the paper compares ordinary control prompts with prompts that ask
the model to write from the inside of different targets: a rock, river, tree,
thermostat, cat, person, all-holding, God, and a later AI hidden-state follow-up.
It also checks whether the same pattern appears when the wording changes, when
the target is external to the model, and when a blind labeling step is used.

## Main Findings

- Expert 114 rises strongly when the model writes in this reflective
  worldview-like style, compared with closely matched control text.
- The signal extends beyond self-reference. It also appears when the model is
  asked to write about external targets such as rocks, rivers, trees, a
  thermostat, all-holding, and God.
- The strongest responses appear for unity-style and theological/all-holding
  prompts. Rock and thermostat prompts also score above the cat prompt in this
  test.
- The AI hidden-state follow-up still activates Expert 114, but at lower
  intensity, landing between cat and river in the prompt ladder.
- The paper reads Expert 114 as a model-specific internal marker for this shift
  in language. It leaves open whether that signal causes the writing style or
  mainly tracks it.

## Zenodo

- Concept DOI: https://doi.org/10.5281/zenodo.20709736
- Current version DOI: https://doi.org/10.5281/zenodo.21048828

## Scope

This repository includes the compiled preprint, LaTeX source, and figure PDFs.
The larger raw run files, model weights, and environment captures live outside
this compact paper package.

## Contents

- `paper/e114_acl_style.pdf` - compiled preprint
- `paper/e114_acl_style.tex` - LaTeX source
- `paper/figs/` - figure PDFs used by the source
- `CITATION.cff` - citation metadata
- `LICENSE` - MIT license
