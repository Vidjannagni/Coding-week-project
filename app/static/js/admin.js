/* ═══════════════════════════════════════════════════════════
   PediAppend – admin.js
   Script spécifique au panneau d'administration (admin.html).
   Gère : bascule du rôle administrateur et suppression
   d'un utilisateur via les endpoints API.
   ═══════════════════════════════════════════════════════════ */

/**
 * Affiche le message d'erreur contenu dans la réponse JSON,
 * ou un message générique si la réponse n'est pas parsable.
 * @param {Response} res - La réponse fetch en erreur.
 */
async function handleApiError(res) {
    let data = {};
    try { data = await res.json(); } catch { /* corps non-JSON ignoré */ }
    alert(data.error || 'Erreur');
}

document.addEventListener('DOMContentLoaded', () => {

    const table = document.querySelector('.admin-table');
    if (!table) return;

    table.addEventListener('click', async (e) => {
        const btn = e.target.closest('button[data-action]');
        if (!btn) return;

        const action = btn.dataset.action;
        const userId = btn.dataset.userId;
        if (!userId) return;

        if (action === 'toggle-admin') {
            if (!confirm('Modifier le rôle de cet utilisateur ?')) return;
            const res = await fetch(`/admin/toggle/${encodeURIComponent(userId)}`, { method: 'POST' });
            if (res.ok) location.reload();
            else await handleApiError(res);
        }

        if (action === 'delete-user') {
            const username = btn.dataset.username || '';
            if (!confirm(`Supprimer l'utilisateur "${username}" et tout son historique ?`)) return;
            const res = await fetch(`/admin/delete/${encodeURIComponent(userId)}`, { method: 'DELETE' });
            if (res.ok) {
                document.getElementById(`user-row-${userId}`)?.remove();
            } else {
                await handleApiError(res);
            }
        }
    });

});
