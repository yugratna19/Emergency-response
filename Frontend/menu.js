document.addEventListener('DOMContentLoaded', function() {
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    let isMenuOpen = false;

    mobileMenuBtn.addEventListener('click', function() {
        isMenuOpen = !isMenuOpen;
        if (isMenuOpen) {
            navLinks.style.display = 'flex';
            mobileMenuBtn.innerHTML = '✕';
        } else {
            navLinks.style.display = 'none';
            mobileMenuBtn.innerHTML = '☰';
        }
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(event) {
        if (isMenuOpen && !event.target.closest('nav')) {
            navLinks.style.display = 'none';
            mobileMenuBtn.innerHTML = '☰';
            isMenuOpen = false;
        }
    });
}); 