/* ═══════════════════════════════════════════════════════════
   PediAppend – result.js
   Script spécifique à la page de résultat (result.html).
   Gère : animation de l'anneau SVG de probabilité
   et animation des barres SHAP.
   ═══════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
    const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ═══ ANIMATION DE L'ANNEAU SVG ═══ */
    /* Anime le cercle de progression et le pourcentage affiché au centre */
    const ringProgress = document.querySelector('.ring-progress');
    const ringPct = document.querySelector('.ring-pct');
    if (ringProgress) {
        const r = 90;
        const c = 2 * Math.PI * r;
        const pct = parseFloat(ringProgress.dataset.pct) || 0;
        ringProgress.style.strokeDasharray = c;
        ringProgress.style.strokeDashoffset = c;
        if (prefersReducedMotion) {
            ringProgress.style.transition = 'none';
            ringProgress.style.strokeDashoffset = c - (pct / 100) * c;
        } else {
            requestAnimationFrame(() => {
                setTimeout(() => {
                    ringProgress.style.transition = 'stroke-dashoffset 1.5s ease-out';
                    ringProgress.style.strokeDashoffset = c - (pct / 100) * c;
                }, 300);
            });
        }
        /* Animation du nombre (comptage de 0 à la valeur cible) */
        if (ringPct) {
            const target = parseFloat(ringPct.dataset.value) || 0;
            if (prefersReducedMotion) {
                ringPct.textContent = target.toFixed(1);
            } else {
                let cur = 0;
                const step = target / 60;
                const timer = setInterval(() => {
                    cur += step;
                    if (cur >= target) { cur = target; clearInterval(timer); }
                    ringPct.textContent = cur.toFixed(1);
                }, 25);
            }
        }
    }

    /* ═══ ANIMATION DES BARRES SHAP ═══ */
    /* Anime la largeur de chaque barre de contribution SHAP */
    document.querySelectorAll('.shap-fill').forEach(bar => {
        const w = bar.dataset.width;
        if (w) {
            if (prefersReducedMotion) bar.style.width = w + '%';
            else setTimeout(() => { bar.style.width = w + '%'; }, 500);
        }
    });

});
