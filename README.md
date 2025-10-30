# ai-vs-humangenerated-images
Build ML models to detect AI-generated vs. real images for the Women in AI x Kaggle Challenge 2025. Using data from Shutterstock and DeepMedia, this project explores fairness, robustness, and ethics in generative AI and deepfake detection.
This project was fascinating, considering how common AI-generated images have become. I can't help but wonder how much of what I see on social media is real. I wanted to understand how a computer "perceives" these fakes and whether I could detect them easily.
Using a dataset from Shutterstock (real images) and DeepMedia (AI-generated counterparts), I set out to train a model that can pick up subtle visual clues that even our eyes might overlook.

## How it works
This project uses PyTorch and RegNetY-8GF, a powerful convolutional model from Torchvision, fine-tuned to classify images into two categories:

0 → Human-made

1 → AI-generated

* The images are resized, flipped, colour-shifted, and rotated for better generalisation.

* The model is pretrained, but I froze the early layers to keep low-level features stable and retrained the classifier head.

* I used AdamW with cosine annealing for smooth optimisation.

* Each epoch, the model’s accuracy, loss, and F1 score are tracked and visualised to monitor progress.

* At the end, it generates a submission.csv with predictions.
