document.addEventListener("DOMContentLoaded", () => {
  const content = document.querySelector(".markdown-body");
  const search = document.querySelector("#table-search");
  const clear = document.querySelector("#clear-search");
  const status = document.querySelector("#search-status");
  const sidebar = document.querySelector(".doc-sidebar");
  const toc = sidebar.querySelector(".toc");
  const nav = content.querySelector(".resource-nav");
  const mobile = window.matchMedia("(max-width: 780px)");
  const normalize = (value) => value.normalize("NFKC").toLowerCase().trim().replace(/\s+/g, " ");

  if (nav) {
    const title = nav.previousElementSibling;
    sidebar.id = title.id;
    title.remove();
    toc.append(nav);
    const shortLabels = [
      "深度学习去噪", "传统去噪", "去噪综述", "去噪与下游任务",
      "EEG 基础模型", "EEG 智能体", "基准评测"
    ];
    const topics = [...nav.querySelectorAll("ol > li:first-child ul a")];
    if (topics.length === shortLabels.length) {
      topics.forEach((link, index) => { link.textContent = shortLabels[index]; });
    }
    sidebar.hidden = false;
    const sizeToc = () => { toc.open = !mobile.matches; };
    sizeToc();
    mobile.addEventListener("change", sizeToc);
  }

  // Keep the Markdown and all of its deep-link targets as the source of truth.
  const sections = [];
  const parents = [];
  let timeline;
  [...content.children].forEach((element) => {
    if (element.matches("h2, h3, h4")) {
      const level = Number(element.tagName[1]);
      while (parents.length && parents.at(-1).level >= level) parents.pop();
      const section = document.createElement("section");
      section.className = "doc-section";
      const parent = parents.at(-1);
      (parent?.section || content).append(section);
      const item = { level, section, parent, heading: element };
      sections.push(item);
      parents.push(item);

      if (element.id === "111-时间线总览") {
        timeline = document.createElement("details");
        timeline.className = "timeline";
        const summary = document.createElement("summary");
        timeline.append(summary);
        summary.append(element);
        section.append(timeline);
      } else {
        section.append(element);
      }
    } else if (parents.length) {
      const { section, heading } = parents.at(-1);
      (heading.id === "111-时间线总览" ? timeline : section).append(element);
    }
  });
  const sectionByElement = new Map(sections.map((item) => [item.section, item]));

  function compactPaper(row) {
    const [number, method, year, idea, title, venue, code, institution, note] = row.cells;
    method.className = "paper-name";
    year.className = "paper-year";
    venue.className = "paper-venue";
    code.className = "paper-links";
    const name = document.createElement("strong");
    const paperTitle = document.createElement("div");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const fields = document.createElement("dl");
    const original = title.querySelector("a");
    name.textContent = method.textContent.trim();
    paperTitle.className = "paper-title";

    if (original && original.textContent.trim() === title.textContent.trim()) {
      paperTitle.textContent = title.textContent.trim();
      original.setAttribute("aria-label", `原文：${paperTitle.textContent}`);
      original.textContent = "原文 ↗";
      original.classList.add("paper-original");
      const availability = code.textContent.trim();
      if (availability === "-" || !availability) code.replaceChildren();
      else if (!code.querySelector("a")) code.prepend("代码：");
      code.prepend(original, document.createTextNode(" "));
    } else {
      paperTitle.append(...title.childNodes);
    }

    summary.textContent = "详情";
    if (name.textContent === "-") name.textContent = "";
    summary.setAttribute("aria-label", `详情：${name.textContent || paperTitle.textContent.trim()}`);
    for (const [label, cell] of [["主要思路", idea], ["作者单位", institution], ["备注", note], ["原序号", number]]) {
      const term = document.createElement("dt");
      const definition = document.createElement("dd");
      term.textContent = label;
      definition.append(...cell.childNodes);
      fields.append(term, definition);
    }
    details.append(summary, fields);
    method.replaceChildren(name, paperTitle, details);
    row.replaceChildren(method, year, venue, code);
  }

  let total = 0;
  const tables = [...content.querySelectorAll("table")].flatMap((table) => {
    const body = table.tBodies[0];
    if (!body) return [];
    const headers = [...table.querySelectorAll("thead th")];
    const section = sectionByElement.get(table.closest(".doc-section"));
    const titles = [];
    for (let item = section; item; item = item.parent) titles.push(item.heading.textContent);
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    table.before(wrapper);
    wrapper.append(table);
    const paper = headers.length === 9 && headers[4].textContent.trim() === "文章名称";
    if (!paper && headers.length >= 7) {
      table.classList.add("wide-table");
      wrapper.tabIndex = 0;
      wrapper.setAttribute("role", "region");
      wrapper.setAttribute("aria-label", `${titles[0] || "表格"}，可左右滚动`);
    }
    const records = [...body.rows].map((row, index) => ({
      row, index,
      values: [...row.cells].map((cell) => cell.textContent.trim()),
      searchText: normalize(`${row.textContent} ${titles.join(" ")}`)
    }));
    total += records.length;
    const columns = paper ? [1, 2, 5, 6] : headers.map((_, index) => index);
    if (paper) {
      table.classList.add("paper-table");
      table.setAttribute("role", "table");
      table.tHead.setAttribute("role", "rowgroup");
      body.setAttribute("role", "rowgroup");
      const headerRow = table.tHead.rows[0];
      headerRow.setAttribute("role", "row");
      headerRow.replaceChildren(...columns.map((index) => headers[index]));
      records.forEach(({ row }) => {
        compactPaper(row);
        row.setAttribute("role", "row");
        [...row.cells].forEach((cell) => cell.setAttribute("role", "cell"));
      });
    }

    function sort(column, direction) {
      headers.forEach((header) => header.removeAttribute("aria-sort"));
      headers[column].setAttribute("aria-sort", direction);
      const sign = direction === "ascending" ? 1 : -1;
      records.sort((a, b) => {
        const left = a.values[column] || "";
        const right = b.values[column] || "";
        const number = (value) => value ? Number(value.replace(/[,%\s]/g, "")) : NaN;
        const [x, y] = [number(left), number(right)];
        const order = Number.isFinite(x) && Number.isFinite(y)
          ? x - y
          : left.localeCompare(right, "zh-CN", { numeric: true, sensitivity: "base" });
        return sign * (order || a.index - b.index);
      }).forEach(({ row }) => body.append(row));
    }
    columns.forEach((column, index) => {
      const header = headers[column];
      header.scope = "col";
      if (paper) header.setAttribute("role", "columnheader");
      const button = document.createElement("button");
      button.type = "button";
      const label = paper ? ["方法 / 论文", "年份", "发表渠道", "资源"][index] : header.textContent.trim();
      button.textContent = label;
      button.setAttribute("aria-label", `${label}：排序`);
      const arrow = document.createElement("span");
      arrow.setAttribute("aria-hidden", "true");
      button.append(arrow);
      button.addEventListener("click", () => {
        sort(column, header.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending");
      });
      header.replaceChildren(button);
    });
    if (paper) sort(2, "descending");
    return [{ records, wrapper, section }];
  });

  const empty = document.createElement("p");
  empty.className = "empty-state";
  empty.textContent = "没有找到匹配的记录，请尝试其他关键词。";
  content.prepend(empty);
  function filter() {
    const query = normalize(search.value);
    const words = query ? query.split(" ") : [];
    const visible = new Set();
    let count = 0;
    for (const { records, wrapper, section } of tables) {
      let matches = 0;
      records.forEach(({ row, searchText }) => {
        row.hidden = !words.every((word) => searchText.includes(word));
        if (!row.hidden) matches++;
      });
      wrapper.hidden = Boolean(query) && matches === 0;
      if (matches) {
        for (let item = section; item; item = item.parent) visible.add(item);
      }
      count += matches;
    }
    sections.forEach((item) => { item.section.hidden = Boolean(query) && !visible.has(item); });
    if (query && timeline && !timeline.parentElement.hidden) timeline.open = true;
    clear.hidden = !query;
    empty.hidden = !query || count > 0;
    status.textContent = query ? `找到 ${count} 条匹配记录` : `共 ${total} 条表格记录 · 点击表头排序`;
  }
  clear.addEventListener("click", () => { search.value = ""; filter(); search.focus(); });
  search.addEventListener("input", filter);
  search.closest(".search-panel").hidden = false;
  filter();

  const headerLinks = [...document.querySelectorAll('.header-nav a[href^="#"]')];
  const localLinks = [...headerLinks, ...nav?.querySelectorAll('a[href^="#"]') || []];
  const fragment = (value) => {
    try { return decodeURIComponent(value.slice(1)); }
    catch { return value.slice(1); }
  };
  const defaultTopic = nav?.querySelector("ul a");
  function updateLocation() {
    const id = fragment(location.hash);
    const target = document.getElementById(id);
    if (target?.closest(".timeline")) timeline.open = true;
    const group = target?.closest(".markdown-body > .doc-section")?.querySelector(":scope > h2");
    const active = id ? localLinks.filter((link) => fragment(link.hash) === id) : [defaultTopic, headerLinks[0]];
    if (group) active.push(headerLinks.find((link) => fragment(link.hash) === group.id));
    localLinks.forEach((link) => {
      if (active.includes(link)) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }
  localLinks.forEach((link) => link.addEventListener("click", () => {
    const target = document.getElementById(fragment(link.hash));
    if (target?.closest("[hidden]")) { search.value = ""; filter(); }
    if (mobile.matches && toc.contains(link)) toc.open = false;
  }));
  window.addEventListener("hashchange", updateLocation);
  updateLocation();
  if (location.hash) requestAnimationFrame(() => document.getElementById(fragment(location.hash))?.scrollIntoView());
});
