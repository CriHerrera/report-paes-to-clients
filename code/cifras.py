import base64
import os

import numpy as np
import pandas as pd


def calcular_matriz_distancias(data):
    df_coords = (
        data[["institution_code", "latitud", "longitud"]]
        .dropna()
        .groupby("institution_code", as_index=False)[["latitud", "longitud"]]
        .mean()
    )
    lat = np.radians(df_coords["latitud"].to_numpy())
    lon = np.radians(df_coords["longitud"].to_numpy())
    lat1 = lat[:, None]
    lon1 = lon[:, None]
    dlat = lat1 - lat
    dlon = lon1 - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist_km = 6371 * c
    return df_coords[["institution_code"]], dist_km


def calcular_promedio_institucion(data):
    data_avg = data.copy()
    data_avg["avg_clec_mate1"] = data_avg[["CLEC_REG_ACTUAL", "MATE1_REG_ACTUAL"]].mean(
        axis=1
    )
    avg_idaux = (
        data_avg.groupby(["institution_code", "ID_aux"])["avg_clec_mate1"]
        .mean()
        .reset_index()
    )
    avg_inst = avg_idaux.groupby("institution_code")["avg_clec_mate1"].mean().reset_index()
    return avg_inst


def calcular_rankings_radio(df_coords, dist_km, avg_inst, radio_km):
    codes = df_coords["institution_code"].to_numpy()
    avg_map = avg_inst.set_index("institution_code")["avg_clec_mate1"]
    avg_vals = avg_map.reindex(codes).to_numpy()
    within = (dist_km <= radio_km) & (dist_km > 0)
    n_neighbors = within.sum(axis=1) + 1
    ranks = []
    for i in range(len(codes)):
        vecinos_idx = np.where(within[i])[0]
        if np.isnan(avg_vals[i]):
            ranks.append(pd.NA)
            continue
        if len(vecinos_idx) == 0:
            ranks.append(1)
            continue
        vecinos_vals = avg_vals[vecinos_idx]
        vecinos_vals = vecinos_vals[~np.isnan(vecinos_vals)]
        if len(vecinos_vals) == 0:
            ranks.append(1)
            continue
        rank_i = 1 + np.sum(vecinos_vals > avg_vals[i])
        ranks.append(int(rank_i))
    df_out = pd.DataFrame(
        {
            "institution_code": codes,
            f"n_institutions_{radio_km}km": n_neighbors,
            f"rank_avg_{radio_km}km": pd.Series(ranks, dtype="Int64"),
        }
    )
    return df_out


def calcular_supera_ponderado(data, ponderaciones):
    df_out = data[["institution_code"]].drop_duplicates().copy()
    for key, info in ponderaciones.items():
        cols = list(info["ponderaciones"].keys())
        pesos = np.array(list(info["ponderaciones"].values()))
        ponderado = data[cols].to_numpy() @ pesos
        supera = ponderado >= info["ultimo_matriculado"]
        counts = (
            data.loc[supera]
            .groupby("institution_code")["ID_aux"]
            .nunique()
            .rename(f"n_supera_{key}")
        )
        df_out = df_out.merge(counts, on="institution_code", how="left")
    return df_out


def preparar_datos(path):
    path_data = os.path.join(path, "inputs", "data")
    data = pd.read_csv(os.path.join(path_data, "ArchivoC_Adm2026REG.csv"), sep=";")
    location = pd.read_csv(os.path.join(path_data, "institutions_location.csv"))
    clientes = pd.read_excel(os.path.join(path_data, "lista_definitiva_clientes.xlsx"))
    comunas = pd.read_csv(os.path.join(path_data, "comunas_chile.csv"))
    all_schools = pd.read_csv(os.path.join(path_data, "institution_code.csv"))

    data = data.rename(columns={"RBD": "institution_code"})
    data["institution_code"] = data["institution_code"].astype(str).str.replace(".0", "")
    data = data.merge(
        location[["institution_code", "latitud", "longitud"]],
        on="institution_code",
        how="left",
    )
    comunas["CODIGO_COMUNA"] = comunas["code_national"].astype(str)
    comunas["CODIGO_COMUNA"] = comunas["CODIGO_COMUNA"].astype(str).str.replace(".0", "")
    data["CODIGO_COMUNA"] = data["CODIGO_COMUNA"].astype(str).str.replace(".0", "")
    data = data.merge(comunas[["CODIGO_COMUNA", "city"]], on="CODIGO_COMUNA", how="left")

    puntajes = [
        "MATE1_REG_ACTUAL",
        "CLEC_REG_ACTUAL",
        "MATE2_REG_ACTUAL",
        "HCSOC_REG_ACTUAL",
        "CIEN_REG_ACTUAL",
    ]
    for puntaje in puntajes:
        data[puntaje] = data[puntaje].replace(0, np.nan)

    clientes["institution_code"] = clientes["institution_code"].astype(str).str.replace(
        ".0", ""
    )
    data = data.merge(
        clientes[["institution_code", "campus_name"]],
        on="institution_code",
        how="left",
    )

    for puntaje in puntajes:
        data[f"max_{puntaje}"] = (data[puntaje] == 1000).astype(int)

    df_institution = data[["institution_code"]].drop_duplicates().copy()
    for puntaje in puntajes:
        max_inst_col = f"max_{puntaje}"
        max_inst = data.groupby("institution_code")[max_inst_col].max()
        df_institution = df_institution.merge(max_inst, on="institution_code", how="left")
    for puntaje in puntajes:
        count_inst = (
            data.groupby("institution_code")[f"max_{puntaje}"]
            .sum()
            .rename(f"n_max_{puntaje}")
        )
        df_institution = df_institution.merge(count_inst, on="institution_code", how="left")
    for puntaje in puntajes:
        p90_inst_col = f"p90_inst_{puntaje}"
        n_inst_col = f"n_p90_inst_{puntaje}"
        p90_inst = data.groupby("institution_code")[puntaje].quantile(0.9).rename(
            p90_inst_col
        )
        df_institution = df_institution.merge(p90_inst, on="institution_code", how="left")
        mask_inst = data[puntaje] >= data.groupby("institution_code")[puntaje].transform(
            "quantile", 0.9
        )
        counts_inst = (
            data.loc[mask_inst]
            .groupby("institution_code")["ID_aux"]
            .nunique()
            .rename(n_inst_col)
        )
        df_institution = df_institution.merge(
            counts_inst, on="institution_code", how="left"
        )
        df_institution.drop(columns=[p90_inst_col], inplace=True)

    df_coords, dist_km = calcular_matriz_distancias(data)
    avg_inst = calcular_promedio_institucion(data)
    df_radio_5 = calcular_rankings_radio(df_coords, dist_km, avg_inst, 5)
    df_radio_10 = calcular_rankings_radio(df_coords, dist_km, avg_inst, 10)

    df_institution = df_institution.merge(df_radio_5, on="institution_code", how="left")
    df_institution = df_institution.merge(df_radio_10, on="institution_code", how="left")

    data_avg = data.copy()
    data_avg["avg_clec_mate1"] = data_avg[["CLEC_REG_ACTUAL", "MATE1_REG_ACTUAL"]].mean(
        axis=1
    )
    p90_comuna = (
        data_avg.groupby("CODIGO_COMUNA")["avg_clec_mate1"]
        .quantile(0.9)
        .rename("p90_comuna")
    )
    data_avg = data_avg.merge(p90_comuna, on="CODIGO_COMUNA", how="left")
    mask_comuna = data_avg["avg_clec_mate1"] >= data_avg["p90_comuna"]
    top10_comuna = (
        data_avg.loc[mask_comuna]
        .groupby("institution_code")["ID_aux"]
        .nunique()
        .rename("n_top10_comuna")
    )
    df_institution = df_institution.merge(top10_comuna, on="institution_code", how="left")
    city_inst = data.groupby("institution_code")["city"].first().rename("city")
    df_institution = df_institution.merge(city_inst, on="institution_code", how="left")
    df_institution = df_institution.merge(all_schools, on="institution_code", how="left")
    ponderaciones = {
        "puc_derecho": {
            "ponderaciones": {
                "PTJE_NEM": 0.2,
                "CLEC_REG_ACTUAL": 0.25,
                "MATE1_REG_ACTUAL": 0.1,
                "PTJE_RANKING": 0.2,
                "HCSOC_REG_ACTUAL": 0.25,
            },
            "ultimo_matriculado": 869.35,
        },
        "puc_medicina": {
            "ponderaciones": {
                "PTJE_NEM": 0.2,
                "CLEC_REG_ACTUAL": 0.15,
                "MATE1_REG_ACTUAL": 0.2,
                "PTJE_RANKING": 0.2,
                "CIEN_REG_ACTUAL": 0.25,
            },
            "ultimo_matriculado": 954.45,
        },
        "puc_ingenieria": {
            "ponderaciones": {
                "PTJE_NEM": 0.2,
                "CLEC_REG_ACTUAL": 0.1,
                "MATE1_REG_ACTUAL": 0.25,
                "PTJE_RANKING": 0.2,
                "CIEN_REG_ACTUAL": 0.15,
                "MATE2_REG_ACTUAL": 0.1,
            },
            "ultimo_matriculado": 899.95,
        },
        "uch_derecho": {
            "ponderaciones": {
                "PTJE_NEM": 0.2,
                "CLEC_REG_ACTUAL": 0.25,
                "MATE1_REG_ACTUAL": 0.1,
                "PTJE_RANKING": 0.2,
                "HCSOC_REG_ACTUAL": 0.25,
            },
            "ultimo_matriculado": 844.9,
        },
        "uch_medicina": {
            "ponderaciones": {
                "PTJE_NEM": 0.1,
                "CLEC_REG_ACTUAL": 0.15,
                "MATE1_REG_ACTUAL": 0.2,
                "PTJE_RANKING": 0.2,
                "CIEN_REG_ACTUAL": 0.35,
            },
            "ultimo_matriculado": 924.80,
        },
        "uch_ingenieria": {
            "ponderaciones": {
                "PTJE_NEM": 0.1,
                "CLEC_REG_ACTUAL": 0.1,
                "MATE1_REG_ACTUAL": 0.2,
                "PTJE_RANKING": 0.25,
                "CIEN_REG_ACTUAL": 0.15,
                "MATE2_REG_ACTUAL": 0.2,
            },
            "ultimo_matriculado": 834.65,
        },
    }

    df_supera = calcular_supera_ponderado(data, ponderaciones)
    df_institution = df_institution.merge(df_supera, on="institution_code", how="left")
    df_institution = df_institution.merge(
        clientes[["institution_code", "campus_name"]],
        on="institution_code",
        how="left",
    )
    return data, df_institution, ponderaciones


def imagen_a_data_uri(image_path):
    if not image_path or not os.path.exists(image_path):
        return ""
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def buscar_logo_institucion(logos_dir, institution_code):
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(logos_dir, f"{institution_code}{ext}")
        if os.path.exists(candidate):
            return candidate
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(logos_dir, str(institution_code), f"logo{ext}")
        if os.path.exists(candidate):
            return candidate
    return ""


def render_reporte(template_path, contexto):
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    for key, value in contexto.items():
        html = html.replace(f"{{{{{key}}}}}", value)
    return html


def formatear_entero(valor, default="0"):
    if pd.isna(valor):
        return default
    return str(int(valor))


def construir_contexto_template(row, fecha_reporte):
    nombre = row.get("institution_name")
    nombre = nombre if isinstance(nombre, str) and nombre else row.get("institution_code")
    comuna = row.get("city")
    comuna = comuna if isinstance(comuna, str) and comuna else "NA"
    contexto = {
        "NOMBRE_DEL_ESTABLECIMIENTO": nombre,
        "NOMBRE_COMUNA": comuna,
        "FECHA_REPORTE": fecha_reporte,
        "CANTIDAD_NACIONALES_M1": formatear_entero(row.get("n_max_MATE1_REG_ACTUAL")),
        "CANTIDAD_NACIONALES_LECTORA": formatear_entero(
            row.get("n_max_CLEC_REG_ACTUAL")
        ),
        "CANTIDAD_NACIONALES_M2": formatear_entero(row.get("n_max_MATE2_REG_ACTUAL")),
        "NUMERO_ALUMNOS_TOP10_COMUNAL": formatear_entero(row.get("n_top10_comuna")),
        "RANKING_5KM": formatear_entero(row.get("rank_avg_5km")),
        "TOTAL_COLEGIOS_5KM": formatear_entero(row.get("n_institutions_5km")),
        "RANKING_10KM": formatear_entero(row.get("rank_avg_10km")),
        "TOTAL_COLEGIOS_10KM": formatear_entero(row.get("n_institutions_10km")),
        "UCH_ING_CIVIL_COUNT": formatear_entero(row.get("n_supera_uch_ingenieria")),
        "UCH_DERECHO_COUNT": formatear_entero(row.get("n_supera_uch_derecho")),
        "UCH_MEDICINA_COUNT": formatear_entero(row.get("n_supera_uch_medicina")),
        "PUC_ING_CIVIL_COUNT": formatear_entero(row.get("n_supera_puc_ingenieria")),
        "PUC_DERECHO_COUNT": formatear_entero(row.get("n_supera_puc_derecho")),
        "PUC_MEDICINA_COUNT": formatear_entero(row.get("n_supera_puc_medicina")),
    }
    return contexto


def main():
    path = r"C:\Users\Cristian Herrera\Documents\GitHub\report-paes-to-clients"
    template_path = os.path.join(path, "inputs", "template", "template_reporte.html")
    logos_dir = os.path.join(path, "inputs", "logos")
    logo_tether_path = os.path.join(logos_dir, "Logo Tether.png")
    output_root = os.path.join(path, "outputs")
    os.makedirs(output_root, exist_ok=True)

    _, df_institution, _ = preparar_datos(path)
    fecha_reporte = pd.Timestamp.today().strftime("%d-%m-%Y")
    df_unique = df_institution.drop_duplicates("institution_code")
    institution_codes_to_render = ["2882"]  # ej: ["12345", "67890"]; vacío = todos
    if institution_codes_to_render:
        df_unique = df_unique[df_unique["institution_code"].isin(institution_codes_to_render)]

    for _, row in df_unique.iterrows():
        institution_code = row["institution_code"]
        out_dir = os.path.join(output_root, str(institution_code))
        os.makedirs(out_dir, exist_ok=True)

        logo_inst_path = buscar_logo_institucion(logos_dir, institution_code)
        logo_institucion = imagen_a_data_uri(logo_inst_path)
        logo_tether = imagen_a_data_uri(logo_tether_path)
        contexto = construir_contexto_template(row, fecha_reporte)
        contexto["LOGO_INSTITUCION"] = logo_institucion
        contexto["LOGO_TETHER"] = logo_tether
        html = render_reporte(template_path, contexto)
        output_path = os.path.join(out_dir, "reporte.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)


if __name__ == "__main__":
    main()

