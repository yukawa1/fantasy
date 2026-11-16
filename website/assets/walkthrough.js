(() => {
  const video = document.getElementById('first-fit-video');
  const section = document.getElementById('app-walkthrough');
  for (const link of section.querySelectorAll('[data-seek]')) {
    link.addEventListener('click', event => {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      video.currentTime = Number(link.dataset.seek);
      video.scrollIntoView({ block: 'center' });
      video.play().catch(() => video.focus());
    });
  }
})();
