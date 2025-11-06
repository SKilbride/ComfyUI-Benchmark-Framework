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
            use_main_workflow_only: bool = False) -> dict:
    if not QT_AVAILABLE:
        print("Qt bindings not found – falling back to CLI mode.")
        sys.exit(0)

    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle("ComfyUI Benchmark – Select Options")
    win.setFixedSize(660, 380)

    layout = QVBoxLayout()
    win.setLayout(layout)

    # ------------------------------------------------------------------
    # ComfyUI folder
    # ------------------------------------------------------------------
    comfy_row = QHBoxLayout()
    comfy_row.addWidget(QLabel("ComfyUI folder (-c):"))
    comfy_edit = QLineEdit(str(comfy_path) if comfy_path else "")
    comfy_row.addWidget(comfy_edit)

    def browse_comfy():
        start_dir = comfy_edit.text().strip() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(win, "Select ComfyUI folder", start_dir)
        if folder:
            comfy_edit.setText(folder)

    btn = QPushButton("Browse…")
    btn.clicked.connect(browse_comfy)
    comfy_row.addWidget(btn)
    layout.addLayout(comfy_row)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------
    workflow_row = QHBoxLayout()
    workflow_row.addWidget(QLabel("Workflow (-w) – ZIP or .json file:"))
    workflow_edit = QLineEdit(str(workflow_path) if workflow_path else "")
    workflow_row.addWidget(workflow_edit)

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

    btn = QPushButton("Browse…")
    btn.clicked.connect(browse_workflow)
    workflow_row.addWidget(btn)
    layout.addLayout(workflow_row)

    # ------------------------------------------------------------------
    # Minimal Extraction
    # ------------------------------------------------------------------
    minimal_check = QCheckBox("Minimal Extraction (Models/Nodes must be previously installed) (-e)")
    minimal_check.setChecked(extract_minimal)
    layout.addWidget(minimal_check)

    # ------------------------------------------------------------------
    # Port
    # ------------------------------------------------------------------
    port_row = QHBoxLayout()
    port_row.addWidget(QLabel("Port (-p):"))
    port_spin = QSpinBox()
    port_spin.setRange(1024, 65535)
    port_spin.setValue(port)
    port_row.addWidget(port_spin)
    port_row.addStretch()
    layout.addLayout(port_row)

    # ------------------------------------------------------------------
    # Generations
    # ------------------------------------------------------------------
    gen_row = QHBoxLayout()
    gen_row.addWidget(QLabel("Number of Image Generations (-g):"))
    generations_spin = QSpinBox()
    generations_spin.setRange(1, 1000)
    generations_spin.setValue(generations)
    gen_row.addWidget(generations_spin)
    gen_row.addStretch()
    layout.addLayout(gen_row)

    # ------------------------------------------------------------------
    # Use Package Defaults
    # ------------------------------------------------------------------
    default_check = QCheckBox("Use Package Defaults (-r)")
    default_check.setChecked(run_default)

    def toggle_gen_spin(checked: bool):
        generations_spin.setEnabled(not checked)

    default_check.toggled.connect(toggle_gen_spin)
    toggle_gen_spin(run_default)
    layout.addWidget(default_check)

    # ------------------------------------------------------------------
    # ADVANCED SECTION (COLLAPSIBLE & HIDDEN BY DEFAULT)
    # ------------------------------------------------------------------
    advanced_group = QGroupBox("Advanced Options")
    advanced_group.setCheckable(True)
    advanced_group.setChecked(False)  # Start collapsed
    advanced_group.setFlat(True)

    # Container for advanced widgets
    advanced_container = QWidget()
    advanced_layout = QFormLayout()
    advanced_container.setLayout(advanced_layout)

    # --- Advanced Options ---
    instances_spin = QSpinBox()
    instances_spin.setRange(1, 100)
    instances_spin.setValue(num_instances)
    advanced_layout.addRow("Concurrent Sessions (-n):", instances_spin)

    extra_args_edit = QLineEdit(" ".join(extra_args) if extra_args else "")
    extra_args_edit.setPlaceholderText("--lowvram --force-cpu etc.")
    advanced_layout.addRow("Extra Args:", extra_args_edit)

    debug_check = QCheckBox("Debug Warmup (--debug-warmup)")
    debug_check.setChecked(debug_warmup)
    advanced_layout.addRow(debug_check)

    no_cleanup_check = QCheckBox("Skip Cleanup (--no-cleanup)")
    no_cleanup_check.setChecked(no_cleanup)
    advanced_layout.addRow(no_cleanup_check)

    main_workflow_check = QCheckBox("Use Main Workflow for Warmup (-u)")
    main_workflow_check.setChecked(use_main_workflow_only)
    advanced_layout.addRow(main_workflow_check)

    # Add container to group
    group_layout = QVBoxLayout()
    group_layout.addWidget(advanced_container)
    group_layout.setContentsMargins(20, 10, 20, 10)
    advanced_group.setLayout(group_layout)

    # Toggle visibility
    def toggle_advanced(checked):
        advanced_container.setVisible(checked)
        win.adjustSize()

    advanced_group.toggled.connect(toggle_advanced)
    toggle_advanced(False)  # Ensure hidden

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

        if not c.exists() or not c.is_dir():
            QMessageBox.critical(win, "Error", "ComfyUI folder does not exist or is not a directory.")
            return
        if not w.exists():
            QMessageBox.critical(win, "Error", "Workflow path does not exist.")
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
            'use_main_workflow_only': main_workflow_check.isChecked()
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