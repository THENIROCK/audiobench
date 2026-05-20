// GoatCounter analytics: cookieless, no personal data, no GDPR banner needed.
// Counts page views, country (from IP, not stored), referrer, browser, OS.
// Sign up at https://www.goatcounter.com/ and put your subdomain below.

(function () {
  var GOATCOUNTER_CODE = "audiobench";
  if (!GOATCOUNTER_CODE || GOATCOUNTER_CODE === "REPLACE_ME") return;

  // Respect Do Not Track / Global Privacy Control.
  if (
    navigator.doNotTrack === "1" ||
    window.doNotTrack === "1" ||
    navigator.globalPrivacyControl === true
  ) {
    return;
  }

  window.goatcounter = { no_onload: true };
  var s = document.createElement("script");
  s.async = true;
  s.src = "//gc.zgo.at/count.js";
  s.setAttribute("data-goatcounter", "https://" + GOATCOUNTER_CODE + ".goatcounter.com/count");
  document.head.appendChild(s);

  function track() {
    if (window.goatcounter && typeof window.goatcounter.count === "function") {
      window.goatcounter.count({ path: location.pathname + location.search + location.hash });
    }
  }

  // mkdocs-material instant navigation: re-fire on every SPA navigation.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(track);
  } else {
    window.addEventListener("load", track);
  }
})();
