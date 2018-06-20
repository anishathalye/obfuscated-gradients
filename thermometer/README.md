# Thermometer Encoding: One Hot Way To Resist Adversarial Examples

Paper: [Buckman et al. 2018](https://openreview.net/forum?id=S18Su--CW)

## Setup

Run `./setup.sh` to fetch models.

## Breaks

* Thermometer Encoding: BPDA

## [robustml] evaluation

Run with:

```bash
python robustml_attack.py --cifar-path <path>
````

[robustml]: https://github.com/robust-ml/robustml
