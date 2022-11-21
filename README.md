# Advanced Methods in Computational Economics

This is an mini-course on "Advanced Methods in Computational
Economics" to solve and estimate dynamic models, held from 14.11.2022 - 21.11.2022 at [CREST](https://crest.science/)/[ENSAE](https://www.ensae.fr/en).

* The lecture suite will take place in person on Monday, 14.11.2022 and on Thursday, 17.11.2022.
* The third lecture on the 21.11.2022 will be held via Zoom
* [Zoom link for 21.11.2022, 13.00 - 16.00](https://unil.zoom.us/j/7056353422)
* Meeting ID: 705 635 3422


## Class enrollment on the [Nuvolos Cloud](https://nuvolos.cloud/)

* All lecture materials (slides, codes, and further readings) will be distributed via the [Nuvolos Cloud](https://nuvolos.cloud/).
* To enroll in this class, please click on this [enrollment key](https://app.nuvolos.cloud/enroll/class/1TU7g4Wz7mk), and follow the steps.


## Purpose of the lectures

* This course is intended to confront M.sc. and Ph.D. students in economics, finance, and related fields with recent tools developed in applied mathematics, machine learning, computational science, and the computational economics literature to solve and estimate (dynamic stochastic) economic models.

* The lectures will be centered around sparse grids, adaptive sparse grids, high-dimensional model representation, two types of machine learning methods (Gaussian Processes and Deep Neural Networks), and will be showcased
in the context of application in macroeconomics, finance, and climate-change economics.

* The lectures will be interactive, in a workshop-like style, that is, a mix of theory and actively playing with code examples (delivered in Python and deployed on a cloud computing infrastructure).


## Prerequisites

* Basic econometrics
* Basic programming in Python (see [this link](https://python-programming.quantecon.org/intro.html) for a thorough Introduction)
* Basic calculus and probability
* [Mathematics for Machine learning](https://mml-book.github.io/) provides a good overview of skills participants are required to be fluent in.


## Topics

#### [Day 1](lectures/lecture_1), Monday, November 14th, 2022 (13.30 - 16.45; Room 2016)

* Introduction to [Sparse Grids](https://github.com/SparseGridsForDynamicEcon/SparseGrids_in_econ_handbook)
* Introduction to [Adaptive Sparse Grids](https://github.com/SparseGridsForDynamicEcon/SparseGrids_in_econ_handbook)
* Introduction to [High-dimensional Model Representation](https://github.com/SparseGridsForDynamicEcon/HDMR)

#### [Day 2](lectures/lecture_2), Thursday, November 17th, 2022 (13.30 - 16.45; Room 2016)

* Deep learning basics
* Multi-layer perceptron
* Feed-forward networks
* Network training - SGD
* Error back-propagation
* Some notes on overfitting
* Throughout lectures - hands-on: Perceptron, gradient descent, Artificial neural networks: a simple MLP implementation and examples of applications
* Introduction to Tensorflow, applied to supervised machine learing
* [Deep Surrogates/Deep Structural Estimation](https://github.com/DeepSurrogate/OptionPricing)
* [Deep Equilibrium Nets](https://github.com/sischei/DeepEquilibriumNets)

#### [Day 3](lectures/lecture_3), Monday, November 21st, 2022 (13.30 - 16.45; Online)
* Basics of Gaussian Process Regression
* Noise-free kernels
* Kernels with noise
* GP classification
* The curse of dimensionality and how to deal with it (e.g., active subspaces)
* Gaussian mixture models (unsupervised machine learning)
* Bayesian active learning
* [Dynamic Programming/optimal control with GPs](https://github.com/GaussianProcessesForDynamicEcon/DynamicIncentiveProblems)
* An outlook to frontier topics of GPs (Limitations of GPs and "big data"/scalable GPs
* [A Gaussian Process Dynamic Programming Code Library](https://github.com/GaussianProcessesForDynamicEcon/DynamicIncentiveProblems)

### Teaching philosophy
Lectures will be interactive, in a workshop-like style,
using [Python](http://www.python.org), [scikit learn](https://scikit-learn.org/), [Tensorflow](https://www.tensorflow.org/), and
[TFP](https://www.tensorflow.org/probability) on [Nuvolos](http://nuvolos.cloud),
a browser-based cloud infrastructure in which files, datasets, code and applications work together,
in order to directly implement and experiment with the introduced methods and algorithms.


### Lecturer
* [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (HEC, University of Lausanne)


### Contacts

- Simon Scheidegger: <simon.scheidegger@unil.ch>
- Nuvolos Support: <support@nuvolos.cloud>


## Evaluation

Participants who take the course for credits are expected to propose a small
project where they apply some of the methods learned to an application (e.g.,
apply deep learning to a data set, solve a dynamic model, etc.). The deliverable
that will be graded is a short write-up (4-6 pages maximum), the data set (if
any), and the code on which the presented results in the report were based. The
participants will have four weeks time after the course is finished to complete
the task. The participants can work alone, or in teams of two.




