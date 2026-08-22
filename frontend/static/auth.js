// Google OAuth Session Handlers
document.addEventListener("DOMContentLoaded", () => {
    // Check login state on page load
    const user = JSON.parse(localStorage.getItem("user"));
    if (user) {
        showLoggedInUser(user);
    }
});

// Callback function triggered after user authenticates with Google
function handleCredentialResponse(response) {
    const idToken = response.credential;

    // Send the token to the FastAPI backend
    fetch("/api/auth/google", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ credential: idToken })
    })
    .then(async (res) => {
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Authentication failed");
        }
        return res.json();
    })
    .then((data) => {
        if (data.success) {
            // Save user profile and app-session token locally
            localStorage.setItem("user", JSON.stringify(data.user));
            localStorage.setItem("token", data.token);

            // Update UI
            showLoggedInUser(data.user);
        }
    })
    .catch((error) => {
        console.error("Login Error:", error);
        alert("Google login failed. Please try again.");
    });
}

function showLoggedInUser(user) {
    const signinContainer = document.getElementById("google-signin-container");
    if (signinContainer) {
        signinContainer.classList.add("hidden");
    }
    
    const profileContainer = document.getElementById("user-profile-container");
    if (profileContainer) {
        profileContainer.classList.remove("hidden");
        profileContainer.classList.add("flex");
    }
    
    const avatar = document.getElementById("user-avatar");
    if (avatar) {
        avatar.src = user.picture;
    }
    
    const nameSpan = document.getElementById("user-name");
    if (nameSpan) {
        nameSpan.textContent = user.name;
    }
}

function handleLogout() {
    // Clear storage
    localStorage.removeItem("user");
    localStorage.removeItem("token");

    // Reset UI and reload
    const signinContainer = document.getElementById("google-signin-container");
    if (signinContainer) {
        signinContainer.classList.remove("hidden");
    }
    
    const profileContainer = document.getElementById("user-profile-container");
    if (profileContainer) {
        profileContainer.classList.add("hidden");
    }
    
    window.location.reload();
}
