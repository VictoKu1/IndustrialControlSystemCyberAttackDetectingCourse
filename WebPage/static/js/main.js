// Get the form
const form = document.getElementById("form");
// Handle the form submit event
form.addEventListener("submit", function(event) {
    event.preventDefault();
    submitForm();
});
// Send a POST request to the server
function submitForm() {
    // Get the form data
    var infoP = document.getElementById("infoP").value;
    // Send a POST request to the server
    fetch("/results", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ infoP: infoP })
        })
        .then(response => {
            if (response.redirected) {
                window.location.href = response.url;
            }
        }).catch(error => { console.log("Error:   " + error); });
}