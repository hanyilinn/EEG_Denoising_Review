document.addEventListener("DOMContentLoaded", () => {
  const searchInput = document.querySelector("#table-search");
  const clearButton = document.querySelector("#clear-search");
  const searchStatus = document.querySelector("#search-status");
  const backToTop = document.querySelector("#back-to-top");
  const content = document.querySelector(".markdown-body");
  const tables = [...content.querySelectorAll("table")];

  let totalRows = 0;
  const emptyState = document.createElement("p");
  emptyState.className = "empty-state";
  emptyState.textContent = "没有找到匹配的记录，请尝试其他关键词。";
  emptyState.hidden = true;
  content.prepend(emptyState);

  const normalize = (value) =>
    value
      .normalize("NFKC")
      .toLocaleLowerCase("zh-CN")
      .replace(/\s+/g, " ")
      .trim();

  const valueForSort = (cell) => {
    const value = normalize(cell.textContent);
    const numeric = Number(value.replace(/[,%\s]/g, ""));
    return value !== "" && Number.isFinite(numeric) ? numeric : value;
  };

  const compareValues = (left, right, direction) => {
    let result;
    if (typeof left === "number" && typeof right === "number") {
      result = left - right;
    } else {
      result = String(left).localeCompare(String(right), "zh-CN", {
        numeric: true,
        sensitivity: "base",
      });
    }
    return direction === "ascending" ? result : -result;
  };

  tables.forEach((table) => {
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);

    const body = table.tBodies[0];
    if (!body) return;

    const rows = [...body.rows];
    totalRows += rows.length;
    rows.forEach((row, originalIndex) => {
      row.dataset.searchText = normalize(row.textContent);
      row.dataset.originalIndex = String(originalIndex);
    });

    [...table.querySelectorAll("thead th")].forEach((header, columnIndex) => {
      header.dataset.sortable = "true";
      header.tabIndex = 0;
      header.setAttribute("role", "button");
      header.setAttribute("aria-label", `${header.textContent.trim()}，点击排序`);

      const sort = () => {
        const direction =
          header.getAttribute("aria-sort") === "ascending"
            ? "descending"
            : "ascending";

        table.querySelectorAll("thead th").forEach((item) => {
          item.removeAttribute("aria-sort");
        });
        header.setAttribute("aria-sort", direction);

        rows
          .sort((leftRow, rightRow) => {
            const leftCell = leftRow.cells[columnIndex];
            const rightCell = rightRow.cells[columnIndex];
            if (!leftCell || !rightCell) return 0;
            return compareValues(
              valueForSort(leftCell),
              valueForSort(rightCell),
              direction
            );
          })
          .forEach((row) => body.appendChild(row));
      };

      header.addEventListener("click", sort);
      header.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          sort();
        }
      });
    });
  });

  const filterRows = () => {
    const query = normalize(searchInput.value);
    let visibleRows = 0;

    tables.forEach((table) => {
      const rows = [...table.tBodies[0].rows];
      let visibleInTable = 0;

      rows.forEach((row) => {
        const matches = query === "" || row.dataset.searchText.includes(query);
        row.hidden = !matches;
        if (matches) visibleInTable += 1;
      });

      table.closest(".table-scroll").hidden = query !== "" && visibleInTable === 0;
      visibleRows += visibleInTable;
    });

    clearButton.hidden = query === "";
    emptyState.hidden = query === "" || visibleRows > 0;
    searchStatus.textContent =
      query === ""
        ? `共收录 ${totalRows} 条表格记录；点击表头可升序或降序排列`
        : `找到 ${visibleRows} 条匹配记录`;
  };

  searchInput.addEventListener("input", filterRows);
  clearButton.addEventListener("click", () => {
    searchInput.value = "";
    filterRows();
    searchInput.focus();
  });

  backToTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  window.addEventListener(
    "scroll",
    () => {
      backToTop.classList.toggle("is-visible", window.scrollY > 700);
    },
    { passive: true }
  );

  filterRows();
});
