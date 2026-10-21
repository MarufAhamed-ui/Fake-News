const form = document.querySelector("#analysis-form");
const textArea = document.querySelector("#news-text");
const count = document.querySelector("#char-count");
const statusLine = document.querySelector("#analysis-status");
const resultRoot = document.querySelector("#analysis-result");
const analyzeButton = document.querySelector("#analyze-button");
const newsStatus = document.querySelector("#news-status");
const newsList = document.querySelector("#news-list");
const refreshButton = document.querySelector("#refresh-news");

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function percentage(value) {
  return (Number(value || 0) * 100).toFixed(1) + "%";
}

function metricPercentage(value) {
  return (Number(value || 0) * 100).toFixed(3) + "%";
}

function createResult(result) {
  const card = el("div", "result-card " + String(result.prediction || "").toLowerCase());
  const top = el("div", "result-top");
  top.append(el("span", "badge", result.prediction || "UNKNOWN"));
  top.append(el("span", "score", "Highest class score: " + percentage(result.score)));
  card.append(top);

  const progress = el("div", "progress");
  const bar = el("span");
  bar.style.width = Math.min(100, Math.max(0, Number(result.score || 0) * 100)) + "%";
  progress.append(bar);
  card.append(progress);
  card.append(el("p", "fine-print", result.reason || ""));

  const columns = el("div", "contributions");
  const groups = [
    ["Terms contributing toward FAKE", result.fake_terms || []],
    ["Terms contributing toward REAL", result.real_terms || []],
  ];
  for (const group of groups) {
    const heading = group[0];
    const terms = group[1];
    const box = el("div", "contribution-box");
    box.append(el("h3", "", heading));
    const list = el("div", "term-list");
    if (terms.length) {
      for (const item of terms) {
        list.append(el("span", "term", item.term + " (" + Number(item.weight).toFixed(3) + ")"));
      }
    } else {
      list.append(el("span", "fine-print", "No strong feature contributions."));
    }
    box.append(list);
    columns.append(box);
  }
  card.append(columns);
  card.append(el("p", "fine-print", "Text contributions explain this model score; they do not prove the article is true or false."));
  return card;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultRoot.replaceChildren();
  statusLine.textContent = "Analyzing…";
  analyzeButton.disabled = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: textArea.value}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The text could not be analyzed.");
    resultRoot.append(createResult(data));
    statusLine.textContent = "";
  } catch (error) {
    statusLine.textContent = error.message || "Could not connect to the analysis service.";
  } finally {
    analyzeButton.disabled = false;
  }
});

textArea.addEventListener("input", () => {
  count.textContent = textArea.value.length.toLocaleString() + " / 25,000";
});

function createNewsCard(article) {
  const card = el("article", "news-card");
  card.append(el("p", "news-meta", (article.source || "News") + " · " + (article.published || "Time not provided")));
  card.append(el("h3", "", article.title || "Headline unavailable"));
  const actions = el("div", "news-actions");
  if (typeof article.link === "string" && article.link.startsWith("https://")) {
    const link = el("a", "", "Open source article");
    link.href = article.link;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    actions.append(link);
  }
  const button = el("button", "text-button", "Analyze headline");
  button.type = "button";
  const output = el("div", "news-result");
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Analyzing…";
    output.replaceChildren();
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({text: article.title || ""}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Analysis failed.");
      output.append(createResult(data));
    } catch (error) {
      output.append(el("p", "status", error.message || "Could not analyze headline."));
    } finally {
      button.disabled = false;
      button.textContent = "Analyze headline";
    }
  });
  actions.append(button);
  card.append(actions, output);
  return card;
}

async function loadNews(refresh) {
  newsStatus.textContent = "Loading headlines…";
  refreshButton.disabled = true;
  newsList.replaceChildren();
  try {
    let url = "/api/news";
    if (refresh) url += "?refresh=1";
    const response = await fetch(url);
    const data = await response.json();
    const errors = data.errors || [];
    for (const error of errors) newsStatus.textContent += " " + error;
    const articles = data.articles || [];
    for (const article of articles) newsList.append(createNewsCard(article));
    if (!articles.length && !errors.length) {
      newsStatus.textContent = "No headlines are available right now.";
    } else if (articles.length) {
      newsStatus.textContent = errors.join(" ");
    }
  } catch (error) {
    newsStatus.textContent = "Could not load RSS headlines. Check the internet connection and refresh.";
  } finally {
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", () => loadNews(true));

function renderMatrix(metrics) {
  const root = document.querySelector("#confusion-matrix");
  const labels = metrics.confusion_matrix_labels || ["REAL", "FAKE"];
  const matrix = metrics.confusion_matrix || [];
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.append(el("th", "", "Actual / Predicted"));
  for (const label of labels) headRow.append(el("th", "", label));
  head.append(headRow);
  table.append(head);
  const body = document.createElement("tbody");
  labels.forEach((label, index) => {
    const row = document.createElement("tr");
    row.append(el("th", "", label));
    for (const value of matrix[index] || []) row.append(el("td", "", Number(value).toLocaleString()));
    body.append(row);
  });
  table.append(body);
  root.replaceChildren(table);
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    const data = await response.json();
    const test = data.test_metrics;
    const root = document.querySelector("#metric-cards");
    if (!test) {
      root.append(el("p", "status", "Training metrics are not available."));
      return;
    }
    const cards = [
      ["Accuracy", test.accuracy],
      ["Macro precision", test.precision_macro],
      ["Macro recall", test.recall_macro],
      ["Macro F1", test.f1_macro],
    ];
    for (const item of cards) {
      const card = el("div", "metric");
      card.append(el("span", "metric-label", item[0]));
      card.append(el("span", "metric-value", metricPercentage(item[1])));
      root.append(card);
    }
    renderMatrix(test);
    document.querySelector("#evaluation-note").textContent = data.evaluation_note || "";
  } catch (error) {
    document.querySelector("#metric-cards").append(el("p", "status", "Could not load metrics."));
  }
}

loadMetrics();
loadNews(false);
