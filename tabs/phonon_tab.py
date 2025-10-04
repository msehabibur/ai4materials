def phonon_tab(pmg_obj: Structure | None):
    st.subheader("🎵 Phonons — default force-field flow (MACE)")
    if pmg_obj is None:
        st.info("Upload/select a structure in the Viewer to compute phonons.")
        return

    c1, c2 = st.columns(2)
    with c1:
        min_length = st.number_input("Supercell min. length (Å)", 8.0, 40.0, 15.0, 1.0, key="ph_min_len")
    with c2:
        store_fc = st.checkbox("Store force constants", value=False, key="ph_store_fc")

    run_btn = st.button("Run Phonon", type="primary", key="ph_run_btn")

    # Show persisted outputs if any
    if st.session_state.get(_PH_DOS_PNG):
        st.image(st.session_state[_PH_DOS_PNG], caption="Phonon DOS (PNG)", use_column_width=True)
    if st.session_state.get(_PH_BS_PNG):
        st.image(st.session_state[_PH_BS_PNG], caption="Phonon Band Structure (PNG)", use_column_width=True)
    if st.session_state.get(_PH_DOS_CSV):
        st.download_button("⬇️ DOS (CSV)",
                           st.session_state[_PH_DOS_CSV], "phonon_dos.csv", key="ph_dl_dos_csv_persist")

    if not run_btn:
        return

    # ---------------- Progress-enabled run ----------------
    # We have 10 logical steps below. Adjust if you add/remove steps.
    _steps = _StepProgress(total_steps=10, label="Phonon workflow running…")

    try:
        _steps.tick("Generating phonon workflow")
        flow = PhononMaker(
            min_length=float(min_length),
            store_force_constants=bool(store_fc)
        ).make(structure=pmg_obj)

        _steps.tick("Launching local run (this may take a while)")
        # Long-running part; we can still advance before/after to show movement
        rs = run_locally(flow, create_folders=True, ensure_success=False)

        _steps.tick("Collecting in-memory results")
        rs_plain = _to_plain(rs)

        _steps.tick("Connecting to JobStore (if configured)")
        store = SETTINGS.JOB_STORE
        if store is not None:
            try:
                store.connect()
            except Exception:
                pass

        _steps.tick("Querying post-processing outputs")
        result = None
        if store is not None:
            candidate_names = [
                "generate_frequencies_eigenvectors",
                "phonon_postprocess",
                "build_phonon_bands",
                "get_frequencies_eigenvectors",
                "generate_phonon_bands",
            ]
            for name in candidate_names:
                try:
                    result = store.query_one(
                        {"name": name},
                        properties=["output.phonon_dos", "output.phonon_bandstructure"],
                        load=True,
                        sort={"completed_at": -1},
                    )
                except Exception:
                    result = None
                if result and isinstance(result, dict):
                    break

        dos_dict = None
        bs_dict  = None
        if result and isinstance(result, dict):
            out = result.get("output") or {}
            if isinstance(out, dict):
                if isinstance(out.get("phonon_dos"), dict):
                    dos_dict = out["phonon_dos"]
                if isinstance(out.get("phonon_bandstructure"), dict):
                    bs_dict = out["phonon_bandstructure"]

        if dos_dict is None and bs_dict is None:
            _steps.tick("Falling back: deep scan of results tree")
            dos_dict, bs_dict = _deep_find(rs_plain)
        else:
            _steps.tick("Found outputs in JobStore")

        if dos_dict is None and bs_dict is None:
            raise RuntimeError(
                "Could not locate phonon outputs. Neither JobStore query nor in-memory scan "
                "found 'phonon_dos' or 'phonon_bandstructure'."
            )

        _steps.tick("Deserializing DOS / bandstructure")
        ph_dos = PhononDos.from_dict(dos_dict) if isinstance(dos_dict, dict) else None
        ph_bs  = PhononBandStructureSymmLine.from_dict(bs_dict) if isinstance(bs_dict, dict) else None

        if ph_dos is None and ph_bs is None:
            raise RuntimeError("Found phonon keys, but could not deserialize DOS or band structure objects.")

        # Plot DOS if available
        _steps.tick("Plotting DOS")
        dos_png = None
        dos_csv = None
        if ph_dos is not None:
            dos_plot = PhononDosPlotter()
            dos_plot.add_dos("Phonon DOS", ph_dos)
            ax_dos = dos_plot.get_plot(); fig_dos = ax_dos.get_figure()
            dos_png = _png_bytes(fig_dos)
            dos_csv = _dos_csv_bytes(ph_dos)
            st.session_state[_PH_DOS_PNG] = dos_png
            st.session_state[_PH_DOS_CSV] = dos_csv

        # Plot Band Structure if available
        _steps.tick("Plotting band structure")
        bs_png = None
        if ph_bs is not None:
            bs_plot = PhononBSPlotter(ph_bs)
            ax_bs   = bs_plot.get_plot(); fig_bs = ax_bs.get_figure()
            bs_png  = _png_bytes(fig_bs)
            st.session_state[_PH_BS_PNG]  = bs_png

        # Optional bundle
        _steps.tick("Storing compact JSON bundle")
        bundle = {
            "has_dos": ph_dos is not None,
            "has_bandstructure": ph_bs is not None,
        }
        st.session_state[_PH_JSON] = json.dumps(bundle).encode()

        _steps.finish(success=True)

        # Render
        st.success("Phonon calculation finished ✅")
        if dos_png:
            st.image(dos_png, caption="Phonon DOS (PNG)", use_column_width=True)
            st.download_button("⬇️ DOS (CSV)", dos_csv, "phonon_dos.csv", key="ph_dl_dos_csv")
        if bs_png:
            st.image(bs_png,  caption="Phonon Band Structure (PNG)", use_column_width=True)

        if (not dos_png) and (not bs_png):
            st.warning("Phonon run completed but no plots were produced (no DOS/BS available).")

    except Exception as e:
        _steps.finish(success=False)
        st.error(f"Phonon workflow failed: {e}")
        with st.expander("Details"):
            st.exception(e)
