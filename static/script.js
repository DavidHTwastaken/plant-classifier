// Function to show image preview upon selection
document.getElementById("file-input").addEventListener("change", function () {
  var file = this.files[0];
  if (file) {
    var reader = new FileReader();
    reader.onload = function (event) {
      var img = document.getElementById("image-preview");
      img.src = event.target.result;
      img.style.display = "block";
    };
    reader.readAsDataURL(file);
  }
});

// Function to upload the image to the server for prediction
document.getElementById("upload-btn").addEventListener("click", function () {
  var fileInput = document.getElementById("file-input");
  var file = fileInput.files[0];
  if (file) {
    sendImageForPrediction(file);
  } else {
    alert("Please select an image to upload.");
  }
});

// Function to send the image data to the backend for prediction
function sendImageForPrediction(imageFile) {
  var formData = new FormData();
  formData.append("image", imageFile);

  fetch("/classify", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((predictions) => displayPredictions(predictions))
    .catch((error) => console.error("Error:", error));
}

// Function to display predictions received from the server
function displayPredictions(predictions) {
  var predictionsList = document.getElementById("predictions-list");
  predictionsList.innerHTML = ""; // Clear previous predictions
  predictions.forEach(function (prediction) {
    var li = document.createElement("li");
    li.textContent =
      prediction.class +
      " (" +
      Math.round(prediction.confidence * 1000) / 100 +
      "%)";
    predictionsList.appendChild(li);
  });
}
