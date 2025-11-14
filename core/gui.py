# core/gui.py
import sys
import os
from pathlib import Path

try:
    from qtpy.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
        QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
        QCheckBox, QSpinBox, QFormLayout
    )
    from qtpy.QtCore import Qt
    QT_AVAILABLE = True
except Exception:  # pragma: no cover
    QT_AVAILABLE = False


def _add_with_margin(parent_layout: QVBoxLayout, child_item) -> None:
    """Add widget OR layout with consistent left margin for alignment.
    
    Auto-detects if child_item is QWidget (addWidget) or QLayout (addLayout).
    """
    margin = QWidget()
    margin.setFixedWidth(12)  # Adjust this to change indent
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(0)
    
    if isinstance(child_item, QWidget):
        # For widgets: margin + widget
        row.addWidget(margin)
        row.addWidget(child_item, stretch=1)
        parent_layout.addLayout(row)
    elif hasattr(child_item, 'addWidget'):  # It's a QLayout (QHBoxLayout, etc.)
        # For layouts: margin + existing layout
        row.addWidget(margin)
        row.addLayout(child_item, stretch=1)
        parent_layout.addLayout(row)
    else:
        raise TypeError(f"Unsupported child_item type: {type(child_item)}")


def run_gui(comfy_path: Path | None = None,
            workflow_path: Path | None = None,
            extract_minimal: bool = False,
            port: int = 8000,
            generations: int = 10,
            num_instances: int = 1,
            run_default: bool = False,
            extra_args: list | None = None,
            debug_warmup: bool = False,
            no_cleanup: bool = False,
            use_main_workflow_only: bool = False,
            override: str | None = None) -> dict:
    if not QT_AVAILABLE:
        print("Qt bindings not found – falling back to CLI mode.")
        sys.exit(0)

    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle("ComfyUI Benchmark – Select Options")
    win.setMinimumSize(660, 280)
    win.resize(660, 280)  # Start size

    layout = QVBoxLayout()
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    win.setLayout(layout)

    # ------------------------------------------------------------------
    # ComfyUI folder
    # ------------------------------------------------------------------
    comfy_row = QHBoxLayout()
    comfy_row.addWidget(QLabel("ComfyUI folder (-c):"))
    comfy_edit = QLineEdit(str(comfy_path) if comfy_path else "")
    comfy_row.addWidget(comfy_edit, 1)

    def browse_comfy():
        start_dir = comfy_edit.text().strip() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(win, "Select ComfyUI folder", start_dir)
        if folder:
            comfy_edit.setText(folder)

    comfy_btn = QPushButton("Browse...")
    comfy_btn.setFixedWidth(90)
    comfy_btn.clicked.connect(browse_comfy)
    comfy_row.addWidget(comfy_btn)

    _add_with_margin(layout, comfy_row)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------
    workflow_row = QHBoxLayout()
    workflow_row.addWidget(QLabel("Workflow (-w) – ZIP or .json file:"))
    workflow_edit = QLineEdit(str(workflow_path) if workflow_path else "")
    workflow_row.addWidget(workflow_edit, 1)

    use_folder_check = QCheckBox("Use Folder (for .json)")
    use_folder_check.setChecked(False)
    workflow_row.addWidget(use_folder_check)

    def browse_workflow():
        start_dir = workflow_edit.text().strip() or os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            win, "Select workflow ZIP or .json", start_dir,
            "Workflow files (*.zip *.json);;All files (*)"
        )
        if file:
            workflow_edit.setText(file)

    workflow_btn = QPushButton("Browse...")
    workflow_btn.setFixedWidth(90)
    workflow_btn.clicked.connect(browse_workflow)
    workflow_row.addWidget(workflow_btn)

    _add_with_margin(layout, workflow_row)

    # ------------------------------------------------------------------
    # Minimal Extraction
    # ------------------------------------------------------------------
    minimal_check = QCheckBox("Minimal Extraction (Models/Nodes must be previously installed) (-e)")
    minimal_check.setChecked(extract_minimal)
    minimal_check.setToolTip("Minimal extraction, only workflow.json, warmup.json, and baseconfig.json are extracted")
    _add_with_margin(layout, minimal_check)

    # ------------------------------------------------------------------
    # Port
    # ------------------------------------------------------------------
    port_row = QHBoxLayout()
    port_row.addWidget(QLabel("Port (-p):"))
    port_spin = QSpinBox()
    port_spin.setToolTip("Port to use for the ComfyUI server")
    port_spin.setRange(1024, 65535)
    port_spin.setValue(port)
    port_row.addWidget(port_spin)
    port_row.addStretch()
    _add_with_margin(layout, port_row)

    # ------------------------------------------------------------------
    # Generations
    # ------------------------------------------------------------------
    gen_row = QHBoxLayout()
    gen_label = QLabel("Number of Generations (-g):")
    gen_label.setToolTip("Number of asset generations to run for each workflow")
    gen_row.addWidget(gen_label)
    generations_spin = QSpinBox()
    generations_spin.setRange(1, 1000)
    generations_spin.setValue(generations)
    generations_spin.setToolTip("Number of asset generations to run for each workflow")
    gen_row.addWidget(generations_spin)
    gen_row.addStretch()
    _add_with_margin(layout, gen_row)

    # ------------------------------------------------------------------
    # Use Package Defaults
    # ------------------------------------------------------------------
    default_check = QCheckBox("Use Package Defaults (-r)")
    default_check.setChecked(run_default)
    default_check.setToolTip("Use package defaults for number of generations and concurrent sessions")

    def toggle_gen_spin(checked: bool):
        generations_spin.setEnabled(not checked)

    default_check.toggled.connect(toggle_gen_spin)
    toggle_gen_spin(run_default)
    _add_with_margin(layout, default_check)

    # ------------------------------------------------------------------
    # ADVANCED SECTION (COLLAPSIBLE & HIDDEN BY DEFAULT)
    # ------------------------------------------------------------------
    advanced_group = QGroupBox("Advanced Options")
    advanced_group.setCheckable(True)
    advanced_group.setChecked(False)
    advanced_group.setFlat(True)

    advanced_container = QWidget()
    advanced_layout = QFormLayout()
    advanced_layout.setContentsMargins(0, 0, 0, 0)
    advanced_container.setLayout(advanced_layout)

    # --- Concurrent Sessions ---
    instances_spin = QSpinBox()
    instances_spin.setRange(1, 100)
    instances_spin.setValue(num_instances)
    advanced_layout.addRow("Concurrent Sessions (-n):", instances_spin)

    # --- Override File (-o) ---
    override_row = QHBoxLayout()
    override_edit = QLineEdit(override or "")
    override_edit.setPlaceholderText("Optional: path to override JSON file")
    override_row.addWidget(override_edit, 1)

    def browse_override():
        start_dir = override_edit.text().strip() or os.getcwd()
        file, _ = QFileDialog.getOpenFileName(
            win, "Select Override JSON File", start_dir,
            "JSON files (*.json);;All files (*)"
        )
        if file:
            override_edit.setText(file)

    override_btn = QPushButton("Browse...")
    override_btn.setFixedWidth(90)
    override_btn.clicked.connect(browse_override)
    override_row.addWidget(override_btn)
    advanced_layout.addRow("Override File (-o):", override_row)

    # --- Extra Args ---
    extra_args_edit = QLineEdit(" ".join(extra_args) if extra_args else "")
    extra_args_edit.setPlaceholderText("--lowvram --force-cpu etc.")
    advanced_layout.addRow("Extra Args:", extra_args_edit)

    # --- Debug Warmup ---
    debug_check = QCheckBox("Debug Warmup (--debug-warmup)")
    debug_check.setChecked(debug_warmup)
    advanced_layout.addRow(debug_check)

    # --- No Cleanup ---
    no_cleanup_check = QCheckBox("Skip Cleanup (--no-cleanup)")
    no_cleanup_check.setChecked(no_cleanup)
    advanced_layout.addRow(no_cleanup_check)

    # --- Use Main Workflow Only ---
    main_workflow_check = QCheckBox("Use Main Workflow for Warmup (-u)")
    main_workflow_check.setChecked(use_main_workflow_only)
    advanced_layout.addRow(main_workflow_check)

    # Add container to group
    group_layout = QVBoxLayout()
    group_layout.addWidget(advanced_container)
    group_layout.setContentsMargins(20, 10, 20, 10)
    advanced_group.setLayout(group_layout)

    # Toggle visibility with dynamic resize
    def toggle_advanced(checked):
        advanced_container.setVisible(checked)
        win.layout().activate()
        win.adjustSize()
        if checked:
            max_h = win.screen().availableGeometry().height() - 100
            if win.height() > max_h:
                win.resize(660, max_h)

    advanced_group.toggled.connect(toggle_advanced)
    toggle_advanced(False)

    layout.addWidget(advanced_group)

    # ------------------------------------------------------------------
    # OK / Cancel
    # ------------------------------------------------------------------
    btn_row = QHBoxLayout()
    btn_row.addStretch()

    ok_btn = QPushButton("OK")
    ok_btn.setDefault(True)

    def on_ok():
        c = Path(comfy_edit.text().strip())
        w = Path(workflow_edit.text().strip())
        use_folder = use_folder_check.isChecked()
        override_path = override_edit.text().strip()

        if not c.exists() or not c.is_dir():
            QMessageBox.critical(win, "Error", "ComfyUI folder does not exist or is not a directory.")
            return
        if not w.exists():
            QMessageBox.critical(win, "Error", "Workflow path does not exist.")
            return
        if override_path and not Path(override_path).exists():
            QMessageBox.critical(win, "Error", "Override file does not exist.")
            return

        if w.suffix.lower() == ".json" and use_folder:
            w = w.parent
        if w.suffix.lower() != ".zip" and not (w / "workflow.json").exists():
            QMessageBox.critical(win, "Error", "Workflow folder must contain a file named 'workflow.json'.")
            return

        win.result = {
            'comfy_path': c,
            'workflow_path': w,
            'extract_minimal': minimal_check.isChecked(),
            'port': port_spin.value(),
            'generations': generations_spin.value(),
            'num_instances': instances_spin.value(),
            'run_default': default_check.isChecked(),
            'extra_args': extra_args_edit.text().strip().split(),
            'debug_warmup': debug_check.isChecked(),
            'no_cleanup': no_cleanup_check.isChecked(),
            'use_main_workflow_only': main_workflow_check.isChecked(),
            'override': override_path or None
        }
        win.close()

    ok_btn.clicked.connect(on_ok)
    btn_row.addWidget(ok_btn)

    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(win.close)
    btn_row.addWidget(cancel_btn)

    layout.addLayout(btn_row)

    win.result = None
    win.show()
    app.exec_()

    if win.result is None:
        print("GUI cancelled – exiting.")
        sys.exit(0)

    return win.result