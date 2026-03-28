from pycaret.datasets import get_data
from pycaret.regression import setup, create_model, save_model

# Load dataset
insurance = get_data("insurance")

# Initialize experiment
setup(
    data=insurance,
    target="charges",
    session_id=123,
    normalize=True,
    polynomial_features=True,
    bin_numeric_features=["age", "bmi"],
)

# Train model
lr = create_model("lr")

# Save pipeline/model
save_model(lr, "model")
