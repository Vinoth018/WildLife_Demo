document.addEventListener('DOMContentLoaded', (event) => {
    const form = document.getElementById('uploadForm');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(form);
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            if (response.ok) {
                window.location.href = "/play";
            } else {
                alert("Failed to upload video.");
            }
        });
    }
});