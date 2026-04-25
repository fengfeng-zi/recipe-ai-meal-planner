(() => {
  const endpointMap = {
    analyze: ["/api/analyze-image", "/analyze-image", "/api/analyze_image"],
    query: ["/api/query", "/query", "/api/ask"],
    plan: ["/api/plan", "/plan", "/api/meal-plan"],
    upload: ["/api/upload-image", "/api/upload", "/upload-image", "/upload", "/api/analyze-image"]
  };

  const mealTypeMap = {
    breakfast: "\u65e9\u9910",
    lunch: "\u5348\u9910",
    dinner: "\u665a\u9910",
    snack: "\u52a0\u9910",
    meal: "\u6b63\u9910"
  };

  const componentLabelMap = {
    sparse_score: "\u7a00\u758f\u53ec\u56de",
    ingredient_overlap: "\u98df\u6750\u91cd\u5408",
    meal_type_match: "\u9910\u522b\u5339\u914d",
    habit_alignment: "\u4e60\u60ef\u5339\u914d",
    log_reuse_bonus: "\u8bb0\u5f55\u590d\u7528",
    vision_alignment: "\u56fe\u50cf\u5bf9\u9f50",
    visual_memory_alignment: "\u89c6\u89c9\u8bb0\u5fc6\u5bf9\u9f50",
    rerank_bonus: "\u91cd\u6392\u52a0\u5206"
  };

  const sourceLabelMap = {
    memory_rag_lab: "RAG \u83dc\u8c31\u5e93",
    catalog: "\u5185\u7f6e\u5019\u9009\u5e93",
    trace: "\u68c0\u7d22\u8ffd\u8e2a",
    unknown: "\u672a\u77e5\u6765\u6e90",
    "\u672a\u77e5\u6765\u6e90": "\u672a\u77e5\u6765\u6e90"
  };

  const el = {
    status: document.getElementById("status"),
    uploadState: document.getElementById("uploadState"),
    uploadProgressBar: document.getElementById("uploadProgressBar") || document.getElementById("uploadProgress"),
    uploadProgressText: document.getElementById("uploadProgressText"),
    sessionId: document.getElementById("sessionId"),
    goal: document.getElementById("goal"),
    planningDays: document.getElementById("planningDays"),
    imagePath: document.getElementById("imagePath"),
    query: document.getElementById("query"),
    allergies: document.getElementById("allergies"),
    dislikes: document.getElementById("dislikes"),
    preferences: document.getElementById("preferences"),
    analysisSummary: document.getElementById("analysisSummary"),
    analysisJson: document.getElementById("analysisJson"),
    retrievalSummary: document.getElementById("retrievalSummary"),
    retrievalJson: document.getElementById("retrievalJson"),
    answerText: document.getElementById("answerText"),
    planSummary: document.getElementById("planSummary"),
    planCards: document.getElementById("planCards"),
    planText: document.getElementById("planText"),
    planJson: document.getElementById("planJson"),
    imageFile: document.getElementById("imageFile") || document.getElementById("imageUpload") || document.getElementById("fileInput"),
    imagePreviewCard: document.getElementById("imagePreviewCard"),
    imagePreview: document.getElementById("imagePreview"),
    imagePreviewCaption: document.getElementById("imagePreviewCaption"),
    evidenceCards: document.getElementById("evidenceCards")
  };

  let latestAnalysis = null;
  let uploadedImagePath = "";
  let previewUrl = "";
  let busyCount = 0;

  const parseCsv = (raw) => String(raw || "").split(",").map((item) => item.trim()).filter(Boolean);
  const fmt = (obj) => JSON.stringify(obj, null, 2);
  const mealTypeLabel = (value) => mealTypeMap[String(value || "").trim().toLowerCase()] || String(value || mealTypeMap.meal);
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => {
    const map = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    };
    return map[char] || char;
  });

  const setStatus = (message, isError = false) => {
    if (!el.status) return;
    el.status.textContent = message;
    el.status.style.color = isError ? "#a03227" : "";
  };

  const setButtonsDisabled = (disabled) => {
    ["analyzeBtn", "queryBtn", "planBtn", "allBtn"].forEach((id) => {
      const node = document.getElementById(id);
      if (node) node.disabled = disabled;
    });
  };

  const withBusy = async (message, task) => {
    busyCount += 1;
    setButtonsDisabled(true);
    if (message) setStatus(message);
    try {
      return await task();
    } finally {
      busyCount = Math.max(0, busyCount - 1);
      if (busyCount === 0) setButtonsDisabled(false);
    }
  };

  const userProfile = () => ({
    allergies: parseCsv(el.allergies && el.allergies.value),
    dislikes: parseCsv(el.dislikes && el.dislikes.value),
    preferences: parseCsv(el.preferences && el.preferences.value)
  });

  const contextPayload = () => ({
    session_id: (el.sessionId && el.sessionId.value.trim()) || "demo",
    goal: (el.goal && el.goal.value.trim()) || "",
    planning_days: Number(el.planningDays && el.planningDays.value) || 5,
    image_path: (el.imagePath && el.imagePath.value.trim()) || uploadedImagePath || "",
    query: (el.query && el.query.value.trim()) || "",
    user_profile: userProfile(),
    image_analysis: latestAnalysis
  });

  const resetUploadProgress = () => {
    if (el.uploadProgressBar) {
      el.uploadProgressBar.value = 0;
      el.uploadProgressBar.hidden = true;
    }
    if (el.uploadProgressText) {
      el.uploadProgressText.textContent = "\u7b49\u5f85\u4e0a\u4f20";
    }
  };

  const updateUploadProgress = (loaded, total) => {
    const hasTotal = Number(total) > 0;
    const ratio = hasTotal ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
    if (el.uploadProgressBar) {
      el.uploadProgressBar.hidden = false;
      el.uploadProgressBar.value = ratio;
    }
    if (el.uploadProgressText) {
      el.uploadProgressText.textContent = hasTotal
        ? `\u5df2\u4e0a\u4f20 ${ratio}%`
        : `\u5df2\u4e0a\u4f20 ${Math.max(1, Math.round(loaded / 1024))} KB`;
    }
  };

  const revokePreviewUrl = () => {
    if (!previewUrl) return;
    URL.revokeObjectURL(previewUrl);
    previewUrl = "";
  };

  const renderLocalPreview = (file) => {
    revokePreviewUrl();
    if (!file || !el.imagePreviewCard || !el.imagePreview) {
      if (el.imagePreviewCard) el.imagePreviewCard.hidden = true;
      return;
    }
    previewUrl = URL.createObjectURL(file);
    el.imagePreview.src = previewUrl;
    el.imagePreviewCard.hidden = false;
    if (el.imagePreviewCaption) {
      const sizeKb = Math.max(1, Math.round(file.size / 1024));
      el.imagePreviewCaption.textContent = `${file.name} · ${sizeKb} KB · \u672c\u5730\u9884\u89c8`;
    }
  };

  const analysisProviderLabel = (analysis, endpoint) => {
    const provider = analysis && analysis.provider ? String(analysis.provider) : "";
    return provider || endpoint || "-";
  };

  const resolveEvidenceSource = (item) => {
    const explicit = item && item.source ? String(item.source) : "";
    if (explicit && explicit !== "unknown") return sourceLabelMap[explicit] || explicit;
    const sourcePath = item && item.sourcePath ? String(item.sourcePath).toLowerCase() : "";
    const chunkId = item && item.chunkId ? String(item.chunkId).toLowerCase() : "";
    const title = item && item.title ? String(item.title).toLowerCase() : "";
    const combined = `${sourcePath} ${chunkId} ${title}`;
    if (combined.includes("multimodal_menu_cards")) return "Multimodal Example";
    if (combined.includes("agent_runtime")) return "Runtime Fixture";
    if (combined.includes("memory_rag_lab")) return "RAG 菜谱库";
    return "未知来源";
  };

  const summarizeAnalysis = (analysis) => {
    if (!analysis) return "\u5c1a\u672a\u8bc6\u522b\u56fe\u7247\u3002";
    const cues = Array.isArray(analysis.recipe_cues) && analysis.recipe_cues.length
      ? analysis.recipe_cues.slice(0, 4).join("\u3001")
      : "\u672a\u63d0\u53d6\u5230\u660e\u786e\u7ebf\u7d22";
    const degraded = analysis.vision_live_failed ? "\uff1b\u5728\u7ebf\u89c6\u89c9\u5931\u8d25\uff0c\u5f53\u524d\u662f\u56de\u9000\u7ed3\u679c" : "";
    return `\u5df2\u8bc6\u522b\uff1a${analysis.dish_name || "\u672a\u77e5\u83dc\u54c1"}\uff1b\u9910\u522b\uff1a${mealTypeLabel(analysis.meal_type)}\uff1b\u70f9\u996a\u65b9\u5f0f\uff1a${analysis.cooking_method || "\u672a\u77e5"}\uff1b\u4efd\u91cf\uff1a${analysis.estimated_portions || 1}\uff1b\u7ebf\u7d22\uff1a${cues}${degraded}`;
  };

  const renderAnalysisDetails = (analysis, endpoint) => {
    if (!analysis) return "\u5c1a\u672a\u751f\u6210\u56fe\u50cf\u8bc6\u522b\u7ed3\u679c\u3002";
    const reconstruction = analysis.recipe_reconstruction && typeof analysis.recipe_reconstruction === "object"
      ? analysis.recipe_reconstruction
      : {};
    const ingredients = Array.isArray(analysis.visible_ingredients) && analysis.visible_ingredients.length
      ? analysis.visible_ingredients.join("\u3001")
      : "\u672a\u8bc6\u522b";
    const nutrition = Array.isArray(analysis.nutrition_signals) && analysis.nutrition_signals.length
      ? analysis.nutrition_signals.join("\u3001")
      : "\u672a\u8bc6\u522b";
    const cautions = Array.isArray(analysis.caution_tags) && analysis.caution_tags.length
      ? analysis.caution_tags.join("\u3001")
      : "\u65e0";
    const cues = Array.isArray(analysis.recipe_cues) && analysis.recipe_cues.length
      ? analysis.recipe_cues.join("\u3001")
      : "\u672a\u63d0\u4f9b";
    const steps = Array.isArray(reconstruction.step_outline) && reconstruction.step_outline.length
      ? reconstruction.step_outline.join("\uff1b")
      : "\u672a\u63d0\u4f9b";
    const substitutions = Array.isArray(reconstruction.substitutions) && reconstruction.substitutions.length
      ? reconstruction.substitutions.join("\u3001")
      : "\u672a\u63d0\u4f9b";
    const diagnostics = analysis.vision_diagnostics && typeof analysis.vision_diagnostics === "object"
      ? analysis.vision_diagnostics
      : null;
    const reason = diagnostics && diagnostics.reason ? String(diagnostics.reason) : "";
    const attempts = diagnostics && Array.isArray(diagnostics.attempts)
      ? diagnostics.attempts.map((item) => `${item.model || "?"}/${item.wire_api || "?"}`).join("、")
      : "";
    return [
      `\u8bc6\u522b\u6765\u6e90\uff1a${analysisProviderLabel(analysis, endpoint)}`,
      `\u83dc\u54c1\u540d\u79f0\uff1a${analysis.dish_name || "\u672a\u77e5"}`,
      `\u9910\u522b\uff1a${mealTypeLabel(analysis.meal_type)}`,
      `\u53ef\u89c1\u98df\u6750\uff1a${ingredients}`,
      `\u8425\u517b\u4fe1\u53f7\uff1a${nutrition}`,
      `\u98ce\u9669\u63d0\u793a\uff1a${cautions}`,
      `\u70f9\u996a\u65b9\u5f0f\uff1a${analysis.cooking_method || "\u672a\u77e5"}`,
      `\u9884\u8ba1\u4efd\u91cf\uff1a${analysis.estimated_portions || 1}`,
      `\u83dc\u8c31\u98ce\u683c\uff1a${reconstruction.dish_style || "\u672a\u63d0\u4f9b"}`,
      `\u5907\u83dc\u6458\u8981\uff1a${reconstruction.prep_summary || analysis.summary || "\u672a\u63d0\u4f9b"}`,
      `\u6b65\u9aa4\u63d0\u7eb2\uff1a${steps}`,
      `\u53ef\u66ff\u6362\u9879\uff1a${substitutions}`,
      `\u68c0\u7d22\u7ebf\u7d22\uff1a${cues}`,
      diagnostics ? `\u5728\u7ebf\u89c6\u89c9\u72b6\u6001\uff1a\u5931\u8d25` : `\u5728\u7ebf\u89c6\u89c9\u72b6\u6001\uff1a\u6b63\u5e38`,
      reason ? `\u5931\u8d25\u539f\u56e0\uff1a${reason}` : `\u5931\u8d25\u539f\u56e0\uff1a\u65e0`,
      attempts ? `\u5c1d\u8bd5\u6a21\u578b\uff1a${attempts}` : `\u5c1d\u8bd5\u6a21\u578b\uff1a\u65e0`
    ].join("\n");
  };

  const ensureEvidenceContainer = () => {
    if (el.evidenceCards) return el.evidenceCards;
    const host = document.createElement("div");
    host.id = "evidenceCards";
    host.className = "evidence-grid";
    if (el.retrievalSummary && el.retrievalSummary.parentElement) {
      el.retrievalSummary.parentElement.insertBefore(host, el.retrievalJson || null);
    }
    el.evidenceCards = host;
    return host;
  };

  const ensurePlanCards = () => {
    if (el.planCards) return el.planCards;
    const host = document.createElement("div");
    host.id = "planCards";
    host.className = "plan-grid plan-days-grid";
    if (el.planText && el.planText.parentElement) {
      el.planText.parentElement.insertBefore(host, el.planText);
    }
    el.planCards = host;
    return host;
  };

  const clearPlanCards = () => {
    const host = ensurePlanCards();
    host.innerHTML = [
      '<article class="plan-day-card empty-card">',
      '<h3>\u8fd8\u6ca1\u6709\u6bcf\u65e5\u8ba1\u5212\u5361\u7247</h3>',
      '<p>\u70b9\u51fb\u201c\u751f\u6210\u81b3\u98df\u8ba1\u5212\u201d\u540e\uff0c\u8fd9\u91cc\u4f1a\u6309\u5929\u663e\u793a\u65e9\u9910\u3001\u5348\u9910\u3001\u665a\u9910\u4e0e\u52a0\u9910\u5efa\u8bae\u3002</p>',
      '</article>'
    ].join("");
    if (el.planSummary) el.planSummary.textContent = "\u5c1a\u672a\u751f\u6210\u81b3\u98df\u8ba1\u5212\u3002";
    if (el.planText) el.planText.hidden = false;
  };

  const renderEvidenceCards = (payload) => {
    const host = ensureEvidenceContainer();
    host.innerHTML = "";

    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    const trace = payload.trace && typeof payload.trace === "object" ? payload.trace : {};
    const reranked = Array.isArray(payload.hits)
      ? payload.hits
      : (Array.isArray(trace.reranked_hits) ? trace.reranked_hits : (Array.isArray(trace.ranked_hits) ? trace.ranked_hits : []));

    const entries = candidates.length
      ? candidates.map((candidate, index) => {
          const hit = reranked[index] || {};
          return {
            title: candidate.title || candidate.name || candidate.chunk_id || `\u5019\u9009 ${index + 1}`,
            mealType: candidate.meal_type || hit.meal_type || "meal",
            score: candidate.score ?? hit.score ?? 0,
            source: candidate.candidate_source || candidate.source || hit.candidate_source || hit.source || "unknown",
            sourcePath: candidate.source_path || hit.source_path || "",
            chunkId: candidate.chunk_id || hit.chunk_id || "",
            ingredients: candidate.ingredients || hit.ingredients || [],
            components: {
              sparse_score: hit.sparse_score,
              ingredient_overlap: hit.ingredient_overlap,
              meal_type_match: hit.meal_type_match,
              habit_alignment: hit.habit_alignment,
              log_reuse_bonus: hit.log_reuse_bonus,
              vision_alignment: hit.vision_alignment,
              visual_memory_alignment: hit.visual_memory_alignment,
              rerank_bonus: hit.rerank_bonus
            }
          };
        })
      : reranked.map((hit, index) => ({
          title: hit.title || hit.name || hit.chunk_id || `\u547d\u4e2d ${index + 1}`,
          mealType: hit.meal_type || "meal",
          score: hit.score ?? 0,
          source: hit.candidate_source || hit.source || "trace",
          sourcePath: hit.source_path || "",
          chunkId: hit.chunk_id || "",
          ingredients: hit.ingredients || [],
          components: {
            sparse_score: hit.sparse_score,
            ingredient_overlap: hit.ingredient_overlap,
            meal_type_match: hit.meal_type_match,
            habit_alignment: hit.habit_alignment,
            log_reuse_bonus: hit.log_reuse_bonus,
            vision_alignment: hit.vision_alignment,
            visual_memory_alignment: hit.visual_memory_alignment,
            rerank_bonus: hit.rerank_bonus
          }
        }));

    if (!entries.length) {
      host.innerHTML = '<div class="empty-card">\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u68c0\u7d22\u8bc1\u636e\u3002</div>';
      return;
    }

    entries.forEach((item) => {
      const card = document.createElement("article");
      card.className = "evidence-card";
      const components = Object.entries(item.components || {})
        .filter(([, value]) => value !== undefined && value !== null && !Number.isNaN(Number(value)) && Number(value) !== 0)
        .map(([name, value]) => `${componentLabelMap[name] || name}: ${Number(value).toFixed(3)}`)
        .join(" | ") || "\u6ca1\u6709\u989d\u5916\u6392\u5e8f\u4fe1\u53f7";
      const ingredientText = Array.isArray(item.ingredients)
        ? item.ingredients.join("\u3001")
        : String(item.ingredients || "").trim();
      const sourceLabel = resolveEvidenceSource(item);
      card.innerHTML = [
        `<h3>${escapeHtml(item.title)}</h3>`,
        `<p class="e-meta">\u9910\u522b\uff1a${escapeHtml(mealTypeLabel(item.mealType))} | \u6765\u6e90\uff1a${escapeHtml(sourceLabel)} | \u5206\u6570\uff1a${Number(item.score || 0).toFixed(3)}</p>`,
        `<p class="e-ing">${ingredientText ? `\u98df\u6750\uff1a${escapeHtml(ingredientText)}` : "\u98df\u6750\uff1a\u672a\u63d0\u4f9b"}</p>`,
        `<p class="e-comp">\u6392\u5e8f\u4f9d\u636e\uff1a${escapeHtml(components)}</p>`
      ].join("");
      host.appendChild(card);
    });
  };

  const callJsonWithFallback = async (kind, payload) => {
    const tried = [];
    for (const endpoint of endpointMap[kind] || []) {
      tried.push(endpoint);
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const text = await response.text();
        let data = {};
        try {
          data = text ? JSON.parse(text) : {};
        } catch {
          data = { raw: text };
        }
        if (!response.ok) continue;
        return { endpoint, data };
      } catch {
      }
    }
    throw new Error(`\u6240\u6709 ${kind} \u63a5\u53e3\u90fd\u5931\u8d25\u4e86\uff1a${tried.join("\u3001")}`);
  };

  const sendFormWithProgress = (endpoint, formData) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", endpoint, true);
    xhr.responseType = "text";
    xhr.upload.onprogress = (event) => updateUploadProgress(event.loaded, event.total);
    xhr.onload = () => {
      let data = {};
      try {
        data = xhr.responseText ? JSON.parse(xhr.responseText) : {};
      } catch {
        data = { raw: xhr.responseText };
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        updateUploadProgress(1, 1);
        resolve({ endpoint, data });
      } else {
        reject(new Error(`\u4e0a\u4f20\u5931\u8d25\uff1a${endpoint} \u8fd4\u56de ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error(`\u4e0a\u4f20\u5931\u8d25\uff1a${endpoint}`));
    xhr.send(formData);
  });

  const callFormWithProgressFallback = async (kind, formData) => {
    const tried = [];
    for (const endpoint of endpointMap[kind] || []) {
      tried.push(endpoint);
      try {
        return await sendFormWithProgress(endpoint, formData);
      } catch {
      }
    }
    throw new Error(`\u6240\u6709 ${kind} \u63a5\u53e3\u90fd\u5931\u8d25\u4e86\uff1a${tried.join("\u3001")}`);
  };

  const normalizeDayCards = (payload) => {
    if (Array.isArray(payload.plan_days) && payload.plan_days.length) return payload.plan_days;

    const planText = payload.plan || payload.answer || payload.final_output || "";
    if (!planText) return [];

    const lines = String(planText).split(/\r?\n/);
    const cards = [];
    let current = null;

    lines.forEach((rawLine) => {
      const line = rawLine.trim();
      if (!line) return;

      const dayMatch = /^(?:Day\s+(\d+)|\u7b2c\s*(\d+)\s*\u5929[:\uff1a]?)$/i.exec(line);
      if (dayMatch) {
        if (current) cards.push(current);
        const dayNumber = Number(dayMatch[1] || dayMatch[2]);
        current = { day: dayNumber, title: `\u7b2c ${dayNumber} \u5929`, meals: [], notes: [] };
        return;
      }

      if (/^(shopping list|\u8d2d\u7269\u6e05\u5355|\u91c7\u8d2d\u6e05\u5355)/i.test(line)) return;
      if (!current) return;

      const mealMatch = /^-\s*([^:\uff1a]+)[:\uff1a]\s*(.+)$/.exec(line);
      if (mealMatch) {
        current.meals.push({ meal: mealMatch[1].trim(), description: mealMatch[2].trim() });
      } else {
        current.notes.push(line);
      }
    });

    if (current) cards.push(current);
    return cards;
  };

  const renderPlanCards = (payload) => {
    const cards = normalizeDayCards(payload);
    const host = ensurePlanCards();
    host.innerHTML = "";

    if (!cards.length) {
      host.innerHTML = [
        '<article class="plan-day-card empty-card">',
        '<h3>\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u6bcf\u65e5\u8ba1\u5212\u5361\u7247</h3>',
        '<p>\u89c4\u5212\u5668\u53ea\u8fd4\u56de\u4e86\u7eaf\u6587\u672c\uff0c\u8bf7\u67e5\u770b\u4e0b\u65b9\u6587\u672c\u7ed3\u679c\u3002</p>',
        '</article>'
      ].join("");
      if (el.planSummary) {
        el.planSummary.textContent = "\u89c4\u5212\u5668\u8fd4\u56de\u4e86\u6587\u672c\u7ed3\u679c\uff0c\u4f46\u6ca1\u6709\u7ed3\u6784\u5316\u7684\u6bcf\u65e5\u8ba1\u5212\u3002";
      }
      if (el.planText) el.planText.hidden = false;
      return false;
    }

    if (el.planSummary) {
      const shoppingCount = Array.isArray(payload.shopping_list) ? payload.shopping_list.length : 0;
      const provider = payload.plan_provider ? `\uff0c\u89c4\u5212\u6765\u6e90\uff1a${payload.plan_provider}` : "";
      el.planSummary.textContent = `\u5df2\u751f\u6210 ${cards.length} \u5929\u7684\u8ba1\u5212\u5361\u7247${shoppingCount ? `\uff0c\u5e76\u63d0\u53d6\u4e86 ${shoppingCount} \u7ec4\u8d2d\u7269\u6e05\u5355` : ""}${provider}\u3002`;
    }

    cards.forEach((day) => {
      const article = document.createElement("article");
      article.className = "plan-day-card meal-day-card";
      const kcalSummary = Array.isArray(day.meals)
        ? day.meals.map((meal) => meal.kcal).filter(Boolean).join(" · ")
        : "";
      const mealsHtml = (day.meals || []).map((meal) => {
        const ingredients = Array.isArray(meal.ingredients) && meal.ingredients.length
          ? `<div class="day-note">\u98df\u6750\uff1a${escapeHtml(meal.ingredients.join("\u3001"))}</div>`
          : "";
        const meta = [meal.kcal, meal.protein, meal.time].filter(Boolean).join(" · ");
        const title = mealTypeLabel(meal.meal) !== meal.meal ? mealTypeLabel(meal.meal) : meal.meal;
        return [
          '<div class="plan-meal day-meal">',
          `<strong>${escapeHtml(title)}</strong>`,
          `<div>${escapeHtml(meal.description || "")}</div>`,
          meta ? `<div class="day-note">${escapeHtml(meta)}</div>` : "",
          ingredients,
          '</div>'
        ].join("");
      }).join("");
      const notesHtml = Array.isArray(day.notes) && day.notes.length
        ? `<div class="plan-note day-note">\u5907\u6ce8\uff1a${escapeHtml(day.notes.join(" "))}</div>`
        : "";
      article.innerHTML = [
        '<div class="plan-day-head day-card-head">',
        `<h3 class="plan-day-title day-title">${escapeHtml(day.title || `\u7b2c ${day.day || "?"} \u5929`)}</h3>`,
        kcalSummary ? `<span class="plan-day-kcal day-kcal">${escapeHtml(kcalSummary)}</span>` : "",
        '</div>',
        `<div class="plan-meals day-meals">${mealsHtml || '<div class="day-meal">\u6ca1\u6709\u89e3\u6790\u5230\u9910\u6b21\u5185\u5bb9\u3002</div>'}</div>`,
        notesHtml
      ].join("");
      host.appendChild(article);
    });

    if (el.planText) el.planText.hidden = true;
    return true;
  };

  const uploadImageIfNeeded = async () => {
    const file = el.imageFile && el.imageFile.files && el.imageFile.files[0] ? el.imageFile.files[0] : null;
    if (!file) return "";

    if (el.uploadState) el.uploadState.textContent = `\u5df2\u9009\u62e9\u56fe\u7247\uff1a${file.name}`;
    resetUploadProgress();

    return withBusy("\u6b63\u5728\u4e0a\u4f20\u56fe\u7247...", async () => {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("session_id", (el.sessionId && el.sessionId.value.trim()) || "demo");
      formData.append("remember", "true");

      const { endpoint, data } = await callFormWithProgressFallback("upload", formData);
      const path = data.image_path || data.uploaded_image_path || data.uploaded_path || "";
      if (!path) {
        throw new Error(`\u4e0a\u4f20\u63a5\u53e3 ${endpoint} \u6ca1\u6709\u8fd4\u56de\u56fe\u7247\u8def\u5f84\u3002`);
      }

      uploadedImagePath = path;
      if (el.imagePath && !el.imagePath.value.trim()) {
        el.imagePath.value = path;
      }

      if (data.analysis && typeof data.analysis === "object") {
        latestAnalysis = data.analysis;
        if (el.analysisSummary) el.analysisSummary.textContent = summarizeAnalysis(latestAnalysis);
        if (el.analysisJson) el.analysisJson.textContent = renderAnalysisDetails(latestAnalysis, endpoint);
      }

      if (el.uploadState) el.uploadState.textContent = `\u56fe\u7247\u5df2\u4e0a\u4f20\uff0c\u670d\u52a1\u7aef\u8def\u5f84\uff1a${path}`;
      if (el.uploadProgressText) el.uploadProgressText.textContent = "\u4e0a\u4f20\u5b8c\u6210";
      return path;
    });
  };

  const resolveImagePath = async () => {
    const typedPath = (el.imagePath && el.imagePath.value.trim()) || "";
    if (typedPath) {
      uploadedImagePath = typedPath;
      return typedPath;
    }
    if (uploadedImagePath) return uploadedImagePath;
    return uploadImageIfNeeded();
  };

  const runAnalyze = async () => {
    const payload = contextPayload();
    payload.image_path = await resolveImagePath();
    if (!payload.image_path) {
      setStatus("\u8bf7\u5148\u9009\u62e9\u4e00\u5f20\u56fe\u7247\uff0c\u6216\u8005\u586b\u5199\u56fe\u7247\u8def\u5f84\u3002", true);
      return;
    }

    await withBusy("\u6b63\u5728\u8bc6\u522b\u56fe\u7247...", async () => {
      const { endpoint, data } = await callJsonWithFallback("analyze", payload);
      latestAnalysis = data.analysis || data.image_analysis || null;
      if (el.analysisSummary) el.analysisSummary.textContent = summarizeAnalysis(latestAnalysis);
      if (el.analysisJson) el.analysisJson.textContent = renderAnalysisDetails(latestAnalysis, endpoint);
      setStatus("\u56fe\u7247\u8bc6\u522b\u5b8c\u6210\u3002");
    });
  };

  const runQuery = async () => {
    const payload = contextPayload();
    const hasImageContext = (el.imagePath && el.imagePath.value.trim()) || (el.imageFile && el.imageFile.files && el.imageFile.files[0]);
    payload.image_path = hasImageContext ? await resolveImagePath() : (payload.image_path || uploadedImagePath);
    if (!payload.query) {
      setStatus("\u8bf7\u5148\u8f93\u5165\u9700\u6c42\u63cf\u8ff0\u3002", true);
      return;
    }

    await withBusy("\u6b63\u5728\u68c0\u7d22\u76f8\u5173\u83dc\u8c31...", async () => {
      const { endpoint, data } = await callJsonWithFallback("query", payload);
      const trace = data.trace && typeof data.trace === "object" ? data.trace : {};
      const evidenceCount = (data.answer_trace && data.answer_trace.evidence_count) || trace.evidence_count || (Array.isArray(data.candidates) ? data.candidates.length : 0);
      const topHits = Array.isArray(trace.reranked_hits)
        ? trace.reranked_hits.length
        : (Array.isArray(trace.ranked_hits) ? trace.ranked_hits.length : (Array.isArray(data.hits) ? data.hits.length : 0));
      if (el.retrievalSummary) {
        const provider = data.answer_provider ? `\u56de\u7b54\u6765\u6e90\uff1a${data.answer_provider}\uff1b` : "";
        el.retrievalSummary.textContent = `\u68c0\u7d22\u5b8c\u6210\uff0c${provider}\u8bc1\u636e\u6570\uff1a${evidenceCount}\uff1b\u9ad8\u5206\u5019\u9009\uff1a${topHits}\uff1b\u63a5\u53e3\uff1a${endpoint}\u3002`;
      }
      renderEvidenceCards(data);
      if (el.retrievalJson) el.retrievalJson.textContent = fmt(data);
      if (el.answerText) el.answerText.textContent = data.answer || "\u5f53\u524d\u6ca1\u6709\u751f\u6210\u63a5\u5730\u56de\u7b54\u3002";
      setStatus("\u68c0\u7d22\u5b8c\u6210\u3002");
    });
  };

  const runPlan = async () => {
    const payload = contextPayload();
    const hasImageContext = (el.imagePath && el.imagePath.value.trim()) || (el.imageFile && el.imageFile.files && el.imageFile.files[0]);
    payload.image_path = hasImageContext ? await resolveImagePath() : (payload.image_path || uploadedImagePath);
    payload.image_paths = payload.image_path ? [payload.image_path] : [];
    payload.allergies = payload.user_profile.allergies;
    payload.dislikes = payload.user_profile.dislikes;
    payload.preferences = payload.user_profile.preferences;
    payload.task = "\u751f\u6210\u81b3\u98df\u8ba1\u5212";
    payload.input_text = payload.query || "\u8bf7\u7ed9\u6211\u4e00\u4efd\u53ef\u6267\u884c\u7684\u81b3\u98df\u8ba1\u5212\u3002";

    await withBusy("\u6b63\u5728\u751f\u6210\u81b3\u98df\u8ba1\u5212...", async () => {
      const { endpoint, data } = await callJsonWithFallback("plan", payload);
      const planText = data.plan || data.answer || data.final_output || "\u5f53\u524d\u6ca1\u6709\u8fd4\u56de\u8ba1\u5212\u6587\u672c\u3002";
      const renderedCards = renderPlanCards(data);
      if (el.planText) {
        el.planText.textContent = planText;
        el.planText.hidden = renderedCards;
      }
      if (el.planJson) el.planJson.textContent = fmt(data);
      setStatus(`\u81b3\u98df\u8ba1\u5212\u5df2\u751f\u6210\uff08\u63a5\u53e3\uff1a${endpoint}\uff09\u3002`);
    });
  };

  const safely = (fn) => async () => {
    try {
      await fn();
    } catch (error) {
      setStatus(String((error && error.message) || error), true);
    }
  };

  if (el.imagePath) {
    el.imagePath.addEventListener("input", () => {
      latestAnalysis = null;
      uploadedImagePath = "";
      resetUploadProgress();
      if (el.uploadState && !el.imagePath.value.trim()) {
        el.uploadState.textContent = "\u5c1a\u672a\u9009\u62e9\u56fe\u7247\u3002";
      }
    });
  }

  if (el.imageFile) {
    el.imageFile.addEventListener("change", () => {
      latestAnalysis = null;
      uploadedImagePath = "";
      resetUploadProgress();
      const file = el.imageFile.files && el.imageFile.files[0] ? el.imageFile.files[0] : null;
      if (el.uploadState) {
        el.uploadState.textContent = file ? `\u5df2\u9009\u62e9\u56fe\u7247\uff1a${file.name}` : "\u5c1a\u672a\u9009\u62e9\u56fe\u7247\u3002";
      }
      renderLocalPreview(file);
      if (el.analysisSummary && !latestAnalysis) {
        el.analysisSummary.textContent = "\u4e0a\u4f20\u56fe\u7247\u540e\uff0c\u8fd9\u91cc\u4f1a\u663e\u793a\u83dc\u54c1\u8bc6\u522b\u7ed3\u679c\u3002";
      }
    });
  }

  const analyzeBtn = document.getElementById("analyzeBtn");
  const queryBtn = document.getElementById("queryBtn");
  const planBtn = document.getElementById("planBtn");
  const allBtn = document.getElementById("allBtn");

  if (analyzeBtn) analyzeBtn.addEventListener("click", safely(runAnalyze));
  if (queryBtn) queryBtn.addEventListener("click", safely(runQuery));
  if (planBtn) planBtn.addEventListener("click", safely(runPlan));
  if (allBtn) {
    allBtn.addEventListener("click", safely(async () => {
      const hasImageContext = (el.imagePath && el.imagePath.value.trim()) || (el.imageFile && el.imageFile.files && el.imageFile.files[0]);
      if (hasImageContext) {
        await runAnalyze();
      }
      await runQuery();
      await runPlan();
    }));
  }

  clearPlanCards();
  resetUploadProgress();
  setStatus("\u51c6\u5907\u5c31\u7eea\u3002");
})();

