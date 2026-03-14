/* ═══════════════════════════════════════════════════════════
   PediAppend – landing.js
   Script spécifique à la page d'accueil (index.html).
   Gère : animation des statistiques, navigation par slides
   (clavier, molette, flèches, points latéraux) et particules.
   ═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
    const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ── Animation de comptage des statistiques ── */
    const statObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.querySelectorAll('.stat-value[data-count]').forEach(el => {
                const target = parseFloat(el.dataset.count);
                if (prefersReducedMotion) {
                    el.textContent = target.toFixed(1);
                } else {
                    const duration = 1500;
                    const start = performance.now();
                    function tick(now) {
                        const progress = Math.min((now - start) / duration, 1);
                        const eased = 1 - Math.pow(1 - progress, 3);
                        el.textContent = (target * eased).toFixed(1);
                        if (progress < 1) requestAnimationFrame(tick);
                    }
                    requestAnimationFrame(tick);
                }
            });
            statObserver.unobserve(entry.target);
        });
    }, { threshold: 0.3 });
    document.querySelectorAll('.stats-grid').forEach(el => statObserver.observe(el));

    /* ── Navigation par slides de la landing page ── */
    const sideSteps = document.querySelectorAll('.side-step');
    const landingSections = document.querySelectorAll('.landing-step');
    let currentSlide = 0;
    let slideAnimating = false;

    function goToSlide(index) {
        if (slideAnimating || index === currentSlide || index < 0 || index >= landingSections.length) return;
        slideAnimating = true;
        if (prefersReducedMotion) {
            const prev = landingSections[currentSlide];
            const next = landingSections[index];
            prev.classList.remove('active', 'slide-exit-up', 'slide-exit-down');
            next.classList.add('active');
            sideSteps.forEach((s, i) => s.classList.toggle('active', i === index));
            next.querySelectorAll('.animate-in').forEach(el => el.classList.add('visible'));
            if (next.querySelector('.stats-grid')) {
                next.querySelectorAll('.stat-value[data-count]').forEach(el => {
                    const target = parseFloat(el.dataset.count);
                    el.textContent = (target || 0).toFixed(1);
                });
            }
            currentSlide = index;
            slideAnimating = false;
            return;
        }
        const goingDown = index > currentSlide;
        const prev = landingSections[currentSlide];
        const next = landingSections[index];

        prev.classList.remove('active');
        prev.classList.add(goingDown ? 'slide-exit-up' : 'slide-exit-down');

        next.style.transition = 'none';
        next.style.transform = goingDown ? 'translateY(40px)' : 'translateY(-40px)';
        next.style.opacity = '0';
        next.classList.add('active');

        void next.offsetHeight;
        next.style.transition = '';
        next.style.transform = '';
        next.style.opacity = '';

        sideSteps.forEach((s, i) => s.classList.toggle('active', i === index));

        next.querySelectorAll('.animate-in').forEach(el => el.classList.add('visible'));

        if (next.querySelector('.stats-grid')) {
            next.querySelectorAll('.stat-value[data-count]').forEach(el => {
                const target = parseFloat(el.dataset.count);
                const duration = 1500, start = performance.now();
                function tick(now) {
                    const progress = Math.min((now - start) / duration, 1);
                    const eased = 1 - Math.pow(1 - progress, 3);
                    el.textContent = (target * eased).toFixed(1);
                    if (progress < 1) requestAnimationFrame(tick);
                }
                requestAnimationFrame(tick);
            });
        }

        currentSlide = index;
        setTimeout(() => {
            prev.classList.remove('slide-exit-up', 'slide-exit-down');
            slideAnimating = false;
        }, 550);
    }

    if (landingSections.length) {
        /* Clic sur les points de navigation latéraux */
        sideSteps.forEach(s => {
            s.addEventListener('click', e => {
                e.preventDefault();
                goToSlide(parseInt(s.dataset.goto));
            });
        });

        /* Clic sur les boutons flèches (précédent/suivant) */
        document.querySelectorAll('.slide-arrow-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (btn.dataset.dir === 'next') goToSlide(currentSlide + 1);
                else goToSlide(currentSlide - 1);
            });
        });

        /* Liens data-goto (ex: bouton "Comment ça marche" du hero) */
        document.querySelectorAll('[data-goto]').forEach(el => {
            if (el.classList.contains('side-step') || el.classList.contains('slide-arrow-btn')) return;
            el.addEventListener('click', e => {
                e.preventDefault();
                goToSlide(parseInt(el.dataset.goto));
            });
        });

        /* Navigation au clavier (flèches haut/bas, gauche/droite) */
        document.addEventListener('keydown', e => {
            if (!document.querySelector('.slides-viewport')) return;
            if (e.key === 'ArrowDown' || e.key === 'ArrowRight') { e.preventDefault(); goToSlide(currentSlide + 1); }
            if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') { e.preventDefault(); goToSlide(currentSlide - 1); }
        });

        /* Navigation à la molette (avec anti-rebond de 800ms) */
        const viewport = document.getElementById('slidesViewport');
        let wheelCooldown = false;
        if (viewport) {
            viewport.addEventListener('wheel', e => {
                e.preventDefault();
                if (wheelCooldown || slideAnimating) return;
                wheelCooldown = true;
                if (e.deltaY > 0) goToSlide(currentSlide + 1);
                else goToSlide(currentSlide - 1);
                setTimeout(() => { wheelCooldown = false; }, 800);
            }, { passive: false });
        }

        /* Initialisation : active la première slide */
        landingSections[0].classList.add('active');
        landingSections[0].querySelectorAll('.animate-in').forEach(el => el.classList.add('visible'));
    }

    /* ── Particules décoratives (si l'élément #particles existe) ── */
    const pc = document.getElementById('particles');
    if (pc && !prefersReducedMotion) {
        for (let i = 0; i < 20; i++) {
            const d = document.createElement('span');
            d.className = 'particle';
            d.style.left = Math.random() * 100 + '%';
            d.style.top = Math.random() * 100 + '%';
            d.style.animationDelay = Math.random() * 6 + 's';
            d.style.animationDuration = (4 + Math.random() * 6) + 's';
            d.style.width = d.style.height = (2 + Math.random() * 4) + 'px';
            pc.appendChild(d);
        }
    }

});
