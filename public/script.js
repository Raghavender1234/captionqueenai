// public/script.js

document.addEventListener('DOMContentLoaded', () => {
  // ---------------------------------------------------------
  // Elements
  // ---------------------------------------------------------
  const form = document.getElementById('unified-form');
  const tierCaptionBtn = document.getElementById('select-caption-tier');
  const tierImageBtn = document.getElementById('select-image-tier');
  const selectedTierInput = document.getElementById('selected-tier');

  const descriptionInput = document.getElementById('post-description');

  const imageFields = document.getElementById('image-fields');
  const imageInput = document.getElementById('image-upload');
  const imageEditsInput = document.getElementById('image-edits');

  const platformSelect = document.getElementById('platform');
  const languageInput = document.getElementById('language-input');

  const loadingSpinner = document.getElementById('loading-spinner');
  const resultsSection = document.getElementById('results');
  const resultsContent = document.getElementById('results-content');

  const captionTierAmount = 5;
  const imageTierAmount = 10;

  let RAZORPAY_KEY_ID_CACHE = null;
  let submitting = false;

  // ---------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------
  const show = (el) => el.classList.remove('hidden');
  const hide = (el) => el.classList.add('hidden');

  function showLoading() {
    hide(resultsSection);
    show(loadingSpinner);
  }

  function showResults() {
    hide(loadingSpinner);
    show(resultsSection);
  }

  function resetResults(msg) {
    resultsContent.innerHTML = msg ? `<p>${msg}</p>` : '';
  }

  function setTier(tier) {
    selectedTierInput.value = tier;

    if (tier === 'image') {
      tierImageBtn.classList.add('active');
      tierImageBtn.setAttribute('aria-selected', 'true');
      tierCaptionBtn.classList.remove('active');
      tierCaptionBtn.setAttribute('aria-selected', 'false');
      hide(imageFields); // briefly hide to avoid layout jump
      // Force reflow then show for smoother transition
      requestAnimationFrame(() => show(imageFields));
    } else {
      tierCaptionBtn.classList.add('active');
      tierCaptionBtn.setAttribute('aria-selected', 'true');
      tierImageBtn.classList.remove('active');
      tierImageBtn.setAttribute('aria-selected', 'false');
      hide(imageFields);
      // Clear image fields when switching back to caption tier
      if (imageInput) imageInput.value = '';
      if (imageEditsInput) imageEditsInput.value = '';
    }
  }

  async function getRazorpayKeyId() {
    if (RAZORPAY_KEY_ID_CACHE) return RAZORPAY_KEY_ID_CACHE;
    const res = await fetch('/config');
    if (!res.ok) throw new Error('Unable to fetch Razorpay Key ID from server.');
    const json = await res.json();
    if (!json?.razorpay_key_id) throw new Error('Server did not return razorpay_key_id.');
    RAZORPAY_KEY_ID_CACHE = json.razorpay_key_id;
    return RAZORPAY_KEY_ID_CACHE;
  }

function validateForm() {
  const tier = selectedTierInput.value;
  if (!descriptionInput.value.trim()) {
    alert('Please describe your photo or post idea.');
    descriptionInput.focus();
    return false;
  }
  if (!platformSelect.value) {
    alert('Please choose a platform.');
    platformSelect.focus();
    return false;
  }
  if (!languageInput.value.trim()) {
    alert('Please enter an output language.');
    languageInput.focus();
    return false;
  }
  if (tier === 'image') {
    if (!imageInput.files || imageInput.files.length === 0) {
      alert('Please upload an image for Image+Edits tier.');
      imageInput.focus();
      return false;
    }
-   if (!imageEditsInput.value.trim()) {
-     alert('Please describe desired edits for Image+Edits tier.');
-     imageEditsInput.focus();
-     return false;
-   }
+   // Edits are optional; backend handles empty edits.
  }
  return true;
}

  // ---------------------------------------------------------
  // Drag & Drop convenience on image dropzone
  // ---------------------------------------------------------
  const dropzone = imageFields?.querySelector('.image-upload-area');
  if (dropzone) {
    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
      dropzone.addEventListener(evt, prevent)
    );
    dropzone.addEventListener('dragover', () => dropzone.classList.add('dragover'));
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      dropzone.classList.remove('dragover');
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        imageInput.files = files;
      }
    });
    // Make the whole area clickable
    dropzone.addEventListener('click', () => imageInput?.click());
    dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        imageInput?.click();
      }
    });
  }

  // ---------------------------------------------------------
  // Tier selection
  // ---------------------------------------------------------
  tierCaptionBtn?.addEventListener('click', () => setTier('caption'));
  tierImageBtn?.addEventListener('click', () => setTier('image'));

  // ---------------------------------------------------------
  // Results rendering + error display (NEW)
  // ---------------------------------------------------------
  const displayResults = (data) => {
    const resultsContent = document.getElementById('results-content');
    resultsContent.innerHTML = ''; // Clear previous results

    // If there's a notice from the server, display it
    if (data.notice) {
      const noticeElement = document.createElement('p');
      noticeElement.className = 'notice';
      noticeElement.textContent = `Notice: ${data.notice}`;
      resultsContent.appendChild(noticeElement);
    }

    // Display generated images if they exist
    if (data.images && data.images.length > 0) {
      const imageContainer = document.createElement('div');
      imageContainer.className = 'image-results';

      data.images.forEach(imageUrl => {
        const imageWrapper = document.createElement('div');
        imageWrapper.className = 'image-wrapper';

        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'Generated Image';

        const downloadLink = document.createElement('a');
        downloadLink.href = imageUrl;
        downloadLink.textContent = 'Download Image';
        downloadLink.className = 'download-button';
        downloadLink.setAttribute('download', ''); // prompts download

        imageWrapper.appendChild(img);
        imageWrapper.appendChild(downloadLink);
        imageContainer.appendChild(imageWrapper);
      });

      resultsContent.appendChild(imageContainer);
    }

    // Display generated captions with copy buttons
    if (data.captions && data.captions.length > 0) {
      const captionList = document.createElement('ul');
      captionList.className = 'caption-list';

      data.captions.forEach(captionText => {
        const listItem = document.createElement('li');

        const textSpan = document.createElement('span');
        textSpan.textContent = captionText;

        const copyButton = document.createElement('button');
        copyButton.textContent = 'Copy';
        copyButton.className = 'copy-button';
        copyButton.onclick = () => {
          navigator.clipboard.writeText(captionText).then(() => {
            copyButton.textContent = 'Copied!';
            setTimeout(() => {
              copyButton.textContent = 'Copy';
            }, 2000);
          });
        };

        listItem.appendChild(textSpan);
        listItem.appendChild(copyButton);
        captionList.appendChild(listItem);
      });

      resultsContent.appendChild(captionList);
    }

    // Nothing returned
    if ((!data.images || data.images.length === 0) &&
        (!data.captions || data.captions.length === 0) &&
        !data.notice) {
      resultsContent.innerHTML = '<p>No results returned.</p>';
    }
  };

  const handleError = (error) => {
    console.error('Error:', error);
    const resultsContent = document.getElementById('results-content');
    resultsContent.innerHTML = `<p class="error">Sorry, something went wrong. Please try again.</p>`;
  };

  // ---------------------------------------------------------
  // Payment + verification + content generation pipeline
  // ---------------------------------------------------------
  async function launchRazorpay(amountInINR, onSuccess) {
    // 0) Ensure Razorpay SDK available
    if (typeof window.Razorpay === 'undefined') {
      alert('Payment system not loaded. Please refresh the page.');
      throw new Error('Razorpay SDK not found.');
    }

    // 1) Create an order on our server
    const orderRes = await fetch('/create-order', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ amount: amountInINR })
    });
    if (!orderRes.ok) {
      const txt = await orderRes.text();
      throw new Error(`Failed to create order: ${txt}`);
    }
    const order = await orderRes.json();

    // 2) Use the same key ID as the server environment
    const keyId = await getRazorpayKeyId();

    const options = {
      key: keyId,
      amount: order.amount,
      currency: order.currency,
      name: 'CaptionAI Service',
      description: `AI Content Generation (₹${amountInINR})`,
      order_id: order.id,
      handler: async function (response) {
        try {
          // 3) Verify payment with our server
          const verifyRes = await fetch('/verify-payment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_order_id: response.razorpay_order_id,
              razorpay_signature: response.razorpay_signature
            })
          });

          if (!verifyRes.ok) {
            const text = await verifyRes.text();
            throw new Error(`Verification failed: ${text}`);
          }

          const verified = await verifyRes.json();
          if (verified?.status !== 'ok') {
            throw new Error('Payment not verified by server.');
          }

          // 4) Proceed to content generation
          await onSuccess();
        } catch (err) {
          console.error(err);
          alert('Payment verification failed. Please try again.');
          resetResults('Payment verification failed. Please try again.');
          showResults();
        }
      },
      prefill: {
        name: 'Test User',
        email: 'test.user@example.com',
        contact: '9999999999'
      },
      notes: { address: 'Razorpay Corporate Office' },
      theme: { color: '#007bff' }
    };

    const rzp = new Razorpay(options);
    rzp.on('payment.failed', function () {
      alert('Payment failed. Please try again.');
      resetResults('Payment failed. Please try again.');
      showResults();
    });
    rzp.open();
  }

  async function runCaptionOnly() {
    const payload = {
      description: descriptionInput.value.trim(),
      platform: platformSelect.value,
      language: languageInput.value.trim()
    };

    try {
      const res = await fetch('/generate-content', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      // if server sends a non-2xx, try to surface it gracefully
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Failed to generate captions.');
      }

      const data = await res.json();
      displayResults(data);
    } catch (e) {
      handleError(e);
    } finally {
      showResults();
    }
  }

  async function runImagePlusEdits() {
    const fd = new FormData();
    fd.append('image', imageInput.files[0]);
    fd.append('description', descriptionInput.value.trim());
    fd.append('edits', imageEditsInput.value.trim());
    fd.append('platform', platformSelect.value);
    fd.append('language', languageInput.value.trim());

    try {
      const res = await fetch('/generate-content', { method: 'POST', body: fd });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || 'Failed to generate edits/captions.');
      }

      const data = await res.json();
      displayResults(data);
    } catch (e) {
      handleError(e);
    } finally {
      showResults();
    }
  }

  // ---------------------------------------------------------
  // Submit flow: validate → pay → verify → generate
  // ---------------------------------------------------------
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (submitting) return;
    if (!validateForm()) return;

    submitting = true;
    showLoading();
    resetResults('');

    const tier = selectedTierInput.value;
    const amount = tier === 'image' ? imageTierAmount : captionTierAmount;

    try {
      await launchRazorpay(
        amount,
        async () => {
          // after verified payment
          if (tier === 'image') {
            await runImagePlusEdits();
          } else {
            await runCaptionOnly();
          }
        }
      );
    } catch (err) {
      console.error(err);
      resetResults('Could not initiate payment. Please try again.');
      showResults();
    } finally {
      submitting = false;
    }
  });
});
