document.addEventListener('DOMContentLoaded', function() {
    // --- 1. Hamburger Menu Toggle ---
    // The toggleMenu function itself is often attached directly to an onclick attribute in HTML
    // (e.g., <div class="hamburger-icon" onclick="toggleMenu()">).
    // If you want to attach it via JS, you'd do it here:
    // const hamburgerIcon = document.querySelector(".hamburger-icon");
    // if (hamburgerIcon) {
    //     hamburgerIcon.addEventListener('click', toggleMenu);
    // }

    // This function can remain global if called via onclick in HTML, or be nested here if you prefer.
    // For clarity with onclick, it's often left global.
    // If you remove onclick from HTML, define it here:
    // function toggleMenu() {
    //     const menu = document.querySelector(".menu-links");
    //     const icon = document.querySelector(".hamburger-icon");
    //     if (menu && icon) {
    //         menu.classList.toggle("open");
    //         icon.classList.toggle("open");
    //     }
    // }


    // --- 2. Chatbot Functionality ---
    const chatForm = document.querySelector("#chat-form");
    const userInput = document.querySelector("#user-input");
    const chatBox = document.querySelector("#chat-box");

    // Only add listener if elements exist (important if chat isn't on every page)
    if (chatForm && userInput && chatBox) {
        chatForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const message = userInput.value.trim();
            if (message === "") return;

            addMessage("You", message);
            userInput.value = "";

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ message }),
                });

                const data = await response.json();
                if (data.reply) {
                    addMessage("Bot", data.reply);
                } else {
                    addMessage("Bot", "⚠️ Error: " + (data.error || "Unknown error"));
                }
            } catch (err) {
                console.error("Chatbot network error:", err); // Log the actual error for debugging
                addMessage("Bot", "⚠️ Network error. Please try again later.");
            }
        });

        function addMessage(sender, text) {
            const msg = document.createElement("div");
            msg.classList.add("message");
            msg.innerHTML = `<strong>${sender}:</strong> ${text}`;
            chatBox.appendChild(msg);
            chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
        }
    }


    // --- 3. Certificate Lightbox Functionality ---
    const thumbnails = document.querySelectorAll('.certificate-thumbnail');
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.querySelector('.close-btn');

    // Only add listeners if lightbox elements exist
    if (lightbox && lightboxImg && closeBtn) {
        thumbnails.forEach(thumbnail => {
            thumbnail.addEventListener('click', function() {
                const fullSrc = this.getAttribute('data-full-src');
                if (fullSrc) { // Ensure data-full-src exists
                    lightbox.style.display = 'flex'; // Show the lightbox (using flex for centering)
                    lightboxImg.src = fullSrc; // Set the full-size image source
                    document.body.style.overflow = 'hidden'; // Prevent background scrolling
                }
            });
        });

        // Close lightbox when close button is clicked
        closeBtn.addEventListener('click', function() {
            closeLightbox();
        });

        // Close lightbox when clicking outside the image (on the overlay itself)
        lightbox.addEventListener('click', function(event) {
            // Check if the click occurred directly on the lightbox container, not on the image or close button
            if (event.target === lightbox || event.target === lightboxImg) { // Also close if image is clicked
                closeLightbox();
            }
        });

        // Close lightbox when Escape key is pressed
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && lightbox.style.display === 'flex') {
                closeLightbox();
            }
        });

        function closeLightbox() {
            lightbox.style.display = 'none'; // Hide the lightbox
            lightboxImg.src = ''; // Clear the image source to save memory/prevent flicker
            document.body.style.overflow = 'auto'; // Re-enable background scrolling
        }
    }
});

// The toggleMenu function should remain global if it's referenced directly in HTML's onclick attributes.
// If you attach it via JavaScript (as commented out above), you can remove this global declaration.
function toggleMenu() {
    const menu = document.querySelector(".menu-links");
    const icon = document.querySelector(".hamburger-icon");
    if (menu && icon) { // Add checks to ensure elements exist
        menu.classList.toggle("open");
        icon.classList.toggle("open");
    }
}