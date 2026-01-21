// Request modifier hook - intercepts outgoing ChatGPT API requests
// and replaces placeholder text with window.__mmsbPendingPrompt

(function() {
  if (window.__mmsbRequestHookInstalled) return;
  window.__mmsbRequestHookInstalled = true;

  console.log('[MMSB RequestHook] Installing request modifier');

  const TARGETS = [
    {origin: "https://chatgpt.com", path: "/backend-api/f/conversation"},
    {origin: "https://chat.openai.com", path: "/backend-api/f/conversation"}
  ];

  window.__mmsbPendingPrompt = window.__mmsbPendingPrompt || null;

  function matchesTarget(input) {
    try {
      const abs = new URL(input, location.href);
      for (const target of TARGETS) {
        if (abs.origin === target.origin && abs.pathname.startsWith(target.path)) {
          return true;
        }
      }
    } catch (err) {
      console.warn('[MMSB RequestHook] URL parse failed', err);
    }
    return false;
  }

  const originalFetch = window.fetch;
  window.fetch = async function(input, init) {
    const isTarget = matchesTarget(typeof input === 'string' ? input : input?.url);
    
    if (isTarget) {
      console.log('[MMSB RequestHook] Intercepted ChatGPT API call');
    }

    // Modify outgoing request if we have pending injection
    if (isTarget && init && typeof init.body === 'string' && window.__mmsbPendingPrompt) {
      try {
        const payload = JSON.parse(init.body);
        console.log('[MMSB RequestHook] Has pending prompt:', window.__mmsbPendingPrompt.substring(0, 50));
        
        // Replace the placeholder text with actual prompt
        if (Array.isArray(payload?.messages) && payload.messages.length > 0) {
          const lastMessage = payload.messages[payload.messages.length - 1];
          if (lastMessage?.content?.parts) {
            console.log('[MMSB RequestHook] Original parts:', lastMessage.content.parts);
            lastMessage.content.parts = [window.__mmsbPendingPrompt];
            console.log('[MMSB RequestHook] Injected prompt:', window.__mmsbPendingPrompt.substring(0, 100) + '...');
          }
        }
        
        init.body = JSON.stringify(payload);
        window.__mmsbPendingPrompt = null;
      } catch (err) {
        console.warn('[MMSB RequestHook] Request modification failed', err);
      }
    }

    return originalFetch(input, init);
  };

  console.log('[MMSB RequestHook] Installed successfully');
})();
