function evaluateCSV() {
    const file = document.getElementById("csv-upload").files[0];

    if (!file) {
        alert("Please upload a CSV file first.");
        return;
    }

    document.getElementById("clv-output").innerText = "Analyzing customer data...";

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict-clv-churn", { // Updated API route
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error("Server returned an error");
        return response.json();
    })
    .then(data => {
        // Display returned JSON data
        // You can customize this depending on what structure your backend returns
        const clvData = JSON.stringify(data, null, 2); // formatted output
        document.getElementById("clv-output").innerText = clvData;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("clv-output").innerText = "❌ Failed to process the file.";
    });
}

document.getElementById("csv-upload").addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        document.getElementById("clv-output").innerText = "File uploaded: " + file.name;
    }
});