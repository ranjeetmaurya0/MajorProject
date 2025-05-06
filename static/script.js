function uploadVideo() {
    const input = document.getElementById('videoFile');
    const file = input.files[0];
  
    if (!file) {
      alert('Please select a video file.');
      return;
    }
  
    const formData = new FormData();
    formData.append('video', file);
  
    fetch('/upload', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      const resultBox = document.getElementById('result');
      const authScore = document.getElementById('authScore');
  
      if (data.authenticity !== undefined) {
        authScore.innerText = `Score: ${data.authenticity}% FAKE`;
      } else {
        authScore.innerText = 'Analysis failed.';
      }
      resultBox.style.display = 'block';
    })
    .catch(error => {
      console.error('Error:', error);
    });
  }
  