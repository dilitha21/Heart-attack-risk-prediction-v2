// ==========================================
// HEART RISK PREDICTION - CLIENT-SIDE LOGIC
// Handles form submission, API calls, and UI updates
// ==========================================

// DOM element references for form and result display
const form = document.getElementById("risk-form");
const riskLevel = document.getElementById("risk-level");
const riskProbability = document.getElementById("risk-probability");
const modelAccuracy = document.getElementById("model-accuracy");
const resultNote = document.getElementById("result-note");
const activityInput = document.getElementById("activity");
const activityValue = document.getElementById("activity-value");

// Get all select elements that need dynamic population from API
const selectFields = Array.from(document.querySelectorAll("select[data-options]"));

/**
 * Populate dropdown select fields with valid options from the API
 * Fetches category options from /options endpoint and dynamically
 * creates option elements for each select field
 */
const populateSelects = async () => {
  // Skip if no select fields need population
  if (selectFields.length === 0) {
    return;
  }

  try {
    // Fetch valid category options from the server
    const response = await fetch("/options");
    if (!response.ok) {
      throw new Error("Options request failed");
    }
    const data = await response.json();

    // Populate each select field with its corresponding options
    selectFields.forEach((select) => {
      const key = select.dataset.options;  // Column name (e.g., "Gender", "Smoking")
      const placeholderText = select.dataset.placeholder || "Select";
      const options = Array.isArray(data[key]) ? data[key] : [];
      
      if (options.length === 0) {
        return;
      }

      // Clear existing options
      select.innerHTML = "";
      
      // Add placeholder option
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = placeholderText;
      placeholder.disabled = true;
      placeholder.selected = true;
      select.appendChild(placeholder);

      // Add each valid option
      options.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
    });
  } catch (error) {
    // Display error message if options can't be loaded
    resultNote.textContent = "Unable to load options. Refresh the page.";
  }
};

// Initialize select fields with options on page load
// Initialize select fields with options on page load
populateSelects();

// Sync physical activity slider value with display
if (activityInput && activityValue) {
  const syncActivityValue = () => {
    activityValue.textContent = activityInput.value;
  };
  // Set initial value
  syncActivityValue();
  // Update display whenever slider moves
  activityInput.addEventListener("input", syncActivityValue);
}

/**
 * Handle form submission and prediction request
 * - Collects form data
 * - Validates numeric inputs
 * - Sends POST request to /predict endpoint
 * - Displays prediction results or error messages
 */
form.addEventListener("submit", async (event) => {
  // Prevent default form submission (page reload)
  event.preventDefault();

  const submitButton = form.querySelector("button[type='submit']");

  // Collect all form field values
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  // List of fields that should be numeric
  const numericKeys = [
    "Age",
    "Cholesterol",
    "Blood Pressure",
    "Physical Activity Level",
  ];

  // Validate and convert numeric fields
  for (const key of numericKeys) {
    const numericValue = Number(payload[key]);
    if (Number.isNaN(numericValue)) {
      // Display error if numeric validation fails
      riskLevel.textContent = "Error";
      riskProbability.textContent = "--";
      modelAccuracy.textContent = "--";
      resultNote.textContent = `Please enter a valid ${key}.`;
      return;
    }
    payload[key] = numericValue;
  }

  // Show loading state while waiting for prediction
  riskLevel.textContent = "Running...";
  riskProbability.textContent = "--";
  modelAccuracy.textContent = "--";
  resultNote.textContent = "Calculating prediction...";
  submitButton.disabled = true;

  try {
    // Send prediction request to server
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    // Handle server error responses
    if (!response.ok) {
      const missingFields = Array.isArray(data.missing)
        ? ` Missing: ${data.missing.join(", ")}.`
        : "";
      throw new Error(`${data.error || "Prediction failed"}.${missingFields}`);
    }

    // Display prediction results
    riskLevel.textContent = data.risk;
    riskProbability.textContent = `${Math.round(data.probability * 100)}%`;
    if (typeof data.accuracy === "number") {
      modelAccuracy.textContent = `${Math.round(data.accuracy * 100)}%`;
    } else {
      modelAccuracy.textContent = "--";
    }
    
    // Apply color coding based on risk level
    const riskLower = data.risk.toLowerCase();
    if (riskLower.includes("high")) {
      riskLevel.style.color = "#dc2626";
      riskProbability.style.color = "#dc2626";
      resultNote.textContent = "⚠️ High risk detected. This is a preliminary assessment for testing purposes only. Please seek immediate advice from a qualified healthcare professional for proper diagnosis and treatment.";
      resultNote.style.background = "#fef2f2";
      resultNote.style.borderColor = "#fecaca";
      resultNote.style.color = "#991b1b";
    } else {
      riskLevel.style.color = "#00C896";
      riskProbability.style.color = "#00C896";
      resultNote.textContent = "ℹ️ This assessment is for testing purposes only. While the results indicate lower risk, please consult with a professional healthcare provider for personalized medical advice and recommendations.";
      resultNote.style.background = "#f0fdf4";
      resultNote.style.borderColor = "#bbf7d0";
      resultNote.style.color = "#166534";
    }
  } catch (error) {
    // Display error message if prediction fails
    riskLevel.textContent = "Error";
    riskProbability.textContent = "--";
    modelAccuracy.textContent = "--";
    resultNote.textContent = error.message;
  } finally {
    // Re-enable submit button
    submitButton.disabled = false;
  }
});
