function sendLocationToServer() {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(function(position) {
        const latitude = position.coords.latitude;
        const longitude = position.coords.longitude;
  
        fetch("/submit_location", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            latitude: latitude,
            longitude: longitude
          })
        })
        .then(response => response.json())
        .then(data => {
          console.log("Location submitted:", data);
        })
        .catch(err => console.error("Failed to send location:", err));
      }, function(error) {
        console.error("Location error:", error);
      });
    } else {
      alert("이 브라우저는 위치 정보를 지원하지 않습니다.");
    }
  }
  
  window.onload = function () {
    sendLocationToServer();
  };
  