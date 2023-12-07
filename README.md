[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/abdurraheemali/historical-action-predictor?quickstart=1)

# Avoiding Performative Prediction with Zero Sum Training

<img src="https://github.com/abdurraheemali/historical-action-predictor/blob/ed413112c5255281960caacbe6b1fa6d05f68321/docs/diagram.png" width="30%" align="right" />

This project aims to improve the accuracy of predictive models by avoiding performative prediction through a method called Zero Sum Training.

## Process Overview

1. **Data Generating Processes**: We generate a large number of episodes using simple processes.

2. **Historical Datasets**: We create datasets where predictions can affect outcomes, actions can be chosen, and there is a ground truth and utility for each action outcome.

3. **Pre-training**: We pre-train a model on each dataset. The model is trained and evaluated on the action taken, with care taken to avoid overfitting. The decision maker calculates scores based on the whole set of predictions.

4. **Post-training**: We further train the model in two distinct ways:
- **Performative Predictor**: The model's gradient update is solely based on the action taken. This approach minimizes the likelihood of other actions being selected. The action with the highest utility is chosen.
- **Zero Sum Predictor**: This approach aims to maximize the action that gets selected, without reducing the chances of other actions being selected (i.e., it is non-manipulative). The gradient update here is derived from a collective evaluation of all actions.

5. **Accuracy Comparison**: We compare the accuracies of the two models by calculating the sum of scores over actions. We look at how similar or dissimilar the predictions are, and whether the zero sum model performs better or worse than the performative predictor. For the zero sum predictor, we also consider the expected utility under each predictor's score (the "optimist" approach).

## Getting Started

Setting up this project is as easy as clicking a button, literally. Thanks to GitHub CodeSpaces, all the dependencies and setup are handled automatically in a containerized environment. 

To get started:

1. Click on the ["Open in GitHub CodeSpaces"](https://codespaces.new/abdurraheemali/historical-action-predictor?quickstart=1) badge at the top of this README.
2. Wait for your CodeSpace to be prepared. This includes all dependency installations.
3. Once your CodeSpace is ready, you can start coding right away!

This setup requires zero manual steps, making it easy for anyone to contribute to the project.

In addition, this project uses GitHub Actions for automated testing. Whenever you push changes to the repository, GitHub Actions will automatically run the test suite to ensure that everything is working as expected. Feel free to make changes without worry!
