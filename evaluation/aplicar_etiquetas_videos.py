"""
Aplica etiquetas true_label al CSV de video claims.
Etiquetado realizado por Claude basado en conocimiento hasta agosto 2025.
Afirmaciones sobre eventos post-agosto-2025 marcadas como NO_VERIFICABLE.
"""

import pandas as pd
from pathlib import Path

CSV = Path(__file__).parent / "video_claims_20260604_011014.csv"

# Orden de procesamiento original (mismo que videos_to_analyze.csv)
VIDEO_ORDER = [
    "https://www.youtube.com/watch?v=WR_SfmcIcSw",
    "https://www.youtube.com/watch?v=Ew-qbwRoXck",
    "https://www.youtube.com/watch?v=ZDvrRmPV2Qs",
    "https://www.youtube.com/watch?v=kLBE44gP1YI",
    "https://www.youtube.com/watch?v=xS7WJqz-3cY",
    "https://www.youtube.com/watch?v=JRs4l7h2nNg",
    "https://www.youtube.com/watch?v=Oc3aANBp4KA",
    "https://www.youtube.com/watch?v=pOkUgDDKgFs",
    "https://www.youtube.com/watch?v=BST-O1Fdx4c",
    "https://www.youtube.com/watch?v=Q6S-BFpYDlM",
    "https://www.youtube.com/watch?v=8urGTdEioOQ",
    "https://www.youtube.com/watch?v=-kGjnN5INHQ",
    "https://www.youtube.com/watch?v=mUv9fzMcXAA",
    "https://www.youtube.com/watch?v=nY9pYrZL70E",
    "https://www.youtube.com/watch?v=qchySE98pUc",
    "https://www.youtube.com/watch?v=dKq02JWtddo",
    "https://www.youtube.com/watch?v=Q--2CdM--dA",
    "https://www.youtube.com/watch?v=wINcA3wA1ME",
    "https://www.youtube.com/watch?v=5sl6VMTT8Qs",
    "https://www.youtube.com/watch?v=3ptSscfgKWU&t=438s",
    "https://www.youtube.com/watch?v=-TTkNU64W8Y",
    "https://www.youtube.com/watch?v=AYD609Lfn_I",
    "https://www.youtube.com/watch?v=fY1xeOjYtyQ",
    "https://www.youtube.com/watch?v=S5ynrmXPuN4",
    "https://www.youtube.com/watch?v=Yi9RylBqfEg",
]

# Etiquetas por índice (1-based, orden del CSV)
# VERDADERO / FALSO / NO_VERIFICABLE
LABELS = {
    # ── NOTICIAS (1-35) ───────────────────────────────────────────────────────
    1:  "NO_VERIFICABLE",  # iPhone 17e $600 — post-conocimiento
    2:  "NO_VERIFICABLE",  # Juicio Twitter/Musk SF — evento específico reciente
    3:  "VERDADERO",       # Qualcomm Snapdragon X Elite — anunciado en 2024
    4:  "VERDADERO",       # Ras Laffan (Qatar) mayor planta GNL del mundo
    5:  "NO_VERIFICABLE",  # Japón 80M barriles reservas — evento específico reciente
    6:  "NO_VERIFICABLE",  # Fed tipos 3.5-3.75% — específico post-conocimiento
    7:  "NO_VERIFICABLE",  # BCE crecimiento 0.9% 2026 — específico post-conocimiento
    8:  "NO_VERIFICABLE",  # Crudo +40-50% estrecho Ormuz — evento reciente
    9:  "NO_VERIFICABLE",  # Gas Australia +23% — evento reciente
    10: "NO_VERIFICABLE",  # Irán tercer exportador urea — no verificable con certeza
    11: "NO_VERIFICABLE",  # Qatar segundo urea, Omán quinto... — rankings específicos
    12: "NO_VERIFICABLE",  # Trump-Petro Casa Blanca — evento reciente
    13: "NO_VERIFICABLE",  # China 61% metales estratégicos — cifra específica
    14: "NO_VERIFICABLE",  # EEUU 15.5% metales — cifra específica
    15: "NO_VERIFICABLE",  # Myanmar 9.4%, Australia 8% — cifras específicas
    16: "VERDADERO",       # Colombia 2do exportador flores (tras Países Bajos)
    17: "NO_VERIFICABLE",  # Xi Jinping recibió Orsi en Pekín — evento reciente
    18: "VERDADERO",       # A400M producido por Airbus en Sevilla (continuidad de CASA)
    19: "NO_VERIFICABLE",  # 50M barriles hacia Houston — evento específico
    20: "VERDADERO",       # ExxonMobil y ConocoPhillips salieron Venezuela 2007
    21: "VERDADERO",       # EEUU sanciones PDVSA enero 2019
    22: "VERDADERO",       # Venezuela mayores reservas crudo del mundo (+300B barriles)
    23: "VERDADERO",       # Groenlandia parte de Dinamarca con recursos en el Ártico
    24: "VERDADERO",       # Ártico se calienta 4 veces más rápido que resto del mundo
    25: "VERDADERO",       # Chevron exención noviembre 2022 para exportar petróleo venezolano
    26: "VERDADERO",       # ~2/3 exportaciones crudo Venezuela a China en 2023
    27: "VERDADERO",       # Chevron única petrolera EEUU autorizada exportar crudo venezolano
    28: "NO_VERIFICABLE",  # Trump sanciones Rosneft/Lukoil — no verificable con certeza
    29: "NO_VERIFICABLE",  # India 68.000 bpd crudo ruso antes invasión — cifra específica
    30: "NO_VERIFICABLE",  # Von der Leyen gas ruso 2028 — declaración específica reciente
    31: "NO_VERIFICABLE",  # UE 4.500M€ gas ruso 1H2024 — cifra específica
    32: "NO_VERIFICABLE",  # Armamento 679B$ 2024 SIPRI — post-conocimiento
    33: "NO_VERIFICABLE",  # México 52B$ remesas 10 meses 2025 — post-conocimiento
    34: "NO_VERIFICABLE",  # Cinco refinerías indias dejaron crudo ruso — reciente
    35: "NO_VERIFICABLE",  # India 1.5M bpd crudo ruso octubre — reciente

    # ── DEPORTES (36-75) ──────────────────────────────────────────────────────
    36: "NO_VERIFICABLE",  # Alexis Vega lesión min 70 — evento reciente específico
    37: "NO_VERIFICABLE",  # Ligue MX tecnología semiautomatizada — anuncio reciente
    38: "VERDADERO",       # UEFA implementó SAOT en Champions League (desde 2022-23)
    39: "FALSO",           # Kansas City Commandos no es equipo real de la NFL
    40: "FALSO",           # Stats Mahomes vs equipo ficticio — partido no existe
    41: "NO_VERIFICABLE",  # Alexis Vega se perderá partido Atlas — evento reciente
    42: "NO_VERIFICABLE",  # Toluca vs América última jornada — calendario reciente
    43: "NO_VERIFICABLE",  # 35 cámaras por estadio Ligue MX — detalle técnico no verificable
    44: "VERDADERO",       # FIFA 2026 tendrá 12 grupos (ampliación a 48 equipos)
    45: "VERDADERO",       # Mundial 2026: 12 grupos de 4 equipos
    46: "VERDADERO",       # Mundial 2026: 48 equipos (desde 32 anteriores)
    47: "VERDADERO",       # 4 bombos de 12 equipos en el sorteo
    48: "VERDADERO",       # Bombo 1: 3 anfitriones + 9 mejores del ranking FIFA
    49: "NO_VERIFICABLE",  # Sudáfrica no llegó al Mundial 2010-2026 — ambiguo
    50: "NO_VERIFICABLE",  # Real Madrid vs Man City 10 diciembre — fecha específica reciente
    51: "VERDADERO",       # Final Copa Mundial 2026 programada para el 19 de julio
    52: "VERDADERO",       # Alemania vs Escocia partido inaugural Euro 2024
    53: "FALSO",           # Euro 1960: final fue URSS 2-1 Yugoslavia (no Yugoslavia vs Francia)
    54: "FALSO",           # Euro 1964: final fue España 2-1 URSS (no Hungría)
    55: "NO_VERIFICABLE",  # Euro 1972 Alemania vs Bélgica 2-1 — datos no suficientemente seguros
    56: "FALSO",           # Euro 1976: Checoslovaquia ganó 3-1 a Países Bajos en prórroga (no 5-4)
    57: "FALSO",           # Euro 1980: Alemania Occidental ganó 1-0 a Checoslovaquia (no al revés)
    58: "FALSO",           # Euro 1984: Francia ganó 2-0 a España en la final (no 1-0 a Dinamarca)
    59: "VERDADERO",       # Euro 1988: partido inaugural Alemania Occidental vs Italia terminó 1-1
    60: "NO_VERIFICABLE",  # Michaela Paestegui bronce paranatación — evento específico
    61: "NO_VERIFICABLE",  # Isaías Zono bronce paranatación — evento específico
    62: "NO_VERIFICABLE",  # Michaela Paestegui categoría S9 — no verificable
    63: "NO_VERIFICABLE",  # Kim Berligarcía bicampeona paranatación — no verificable
    64: "NO_VERIFICABLE",  # Kim Berligarcía 20km Europa 2024 — específico
    65: "NO_VERIFICABLE",  # Valeria Sándiga Panamericano Jalapa — específico
    66: "NO_VERIFICABLE",  # Valeria Sándiga cuatro medallas Francia — específico
    67: "NO_VERIFICABLE",  # Valeria Sándiga medallas ciclismo pista — específico
    68: "VERDADERO",       # Era abierta tenis comenzó 1968
    69: "VERDADERO",       # Rod Laver Grand Slam 1962 (amateur) y 1969 (profesional)
    70: "VERDADERO",       # Federer 237 semanas consecutivas en el #1 (feb 2004 - ago 2008)
    71: "VERDADERO",       # Pete Sampras se retiró en 2002
    72: "FALSO",           # Sampras NO fue #1 seis años CONSECUTIVOS (tuvo interrupciones)
    73: "VERDADERO",       # Sampras 14 Grand Slams, récord mundial en su retiro en 2002
    74: "VERDADERO",       # Tenista español (Nadal) a los 35 años tenía 20 Grand Slams
    75: "VERDADERO",       # Nadal ganó su 13er Roland Garros en 2020

    # ── CIENCIA (76-115) ──────────────────────────────────────────────────────
    76:  "VERDADERO",       # Física cuántica estudia naturaleza atómica/subatómica
    77:  "VERDADERO",       # Electrones, protones y neutrones son componentes del átomo
    78:  "VERDADERO",       # Relojes atómicos utilizan átomos para medir el tiempo
    79:  "VERDADERO",       # Medir en física cuántica altera el estado del sistema
    80:  "VERDADERO",       # Quarks y gluones son componentes fundamentales de la materia
    81:  "VERDADERO",       # José Ignacio Latorre es físico español
    82:  "VERDADERO",       # Física cuántica permitió desarrollo del láser
    83:  "VERDADERO",       # Física cuántica permitió desarrollo de la resonancia magnética
    84:  "VERDADERO",       # Luz de la Luna tarda ~1,3 s en llegar a la Tierra (~384.400 km)
    85:  "VERDADERO",       # Luz del Sol tarda ~8 minutos en llegar a la Tierra
    86:  "VERDADERO",       # Luz de Próxima Centauri tarda ~4 años en llegar a la Tierra
    87:  "VERDADERO",       # Velocidad de la luz ~1.000 millones km/h (≈1.080.000.000)
    88:  "VERDADERO",       # Superficie de última dispersión ~380.000 años tras el Big Bang
    89:  "VERDADERO",       # Luz visible humana entre 400 y 700 nm (rango aproximado correcto)
    90:  "VERDADERO",       # Edgar Allan Poe escribió "Eureka" (1848)
    91:  "VERDADERO",       # Poe fue escritor y aficionado a la cosmología/astronomía
    92:  "VERDADERO",       # Rover Perseverance explora cráter Jezero desde 2021
    93:  "VERDADERO",       # NASA y ESA planean Mars Sample Return
    94:  "VERDADERO",       # Cráter Jezero fue lago con delta hace miles de millones de años
    95:  "VERDADERO",       # Minerales pueden formarse por procesos abióticos
    96:  "VERDADERO",       # Perseverance perforó roca en Jezero en 2023 y extrajo muestra
    97:  "VERDADERO",       # Muestras de Jezero contienen minerales como arcilla y carbonatos
    98:  "VERDADERO",       # Muestra de Jezero contiene carbono orgánico, fósforo y azufre
    99:  "NO_VERIFICABLE",  # Bivianita y greijita en manchas leopardo — nombres muy específicos
    100: "VERDADERO",       # Primer exoplaneta alrededor de estrella tipo Sol descubierto hace ~30 años (51 Peg b, 1995)
    101: "VERDADERO",       # Más de 6.000 exoplanetas descubiertos (superado en 2024)
    102: "VERDADERO",       # Plantas y fitoplancton producen oxígeno mediante fotosíntesis
    103: "VERDADERO",       # ~75% exoplanetas detectados por método del tránsito (Kepler/TESS)
    104: "VERDADERO",       # Estrella de Teegarden a 12,5 años luz de la Tierra
    105: "VERDADERO",       # Velocidad radial ~20% exoplanetas; efecto Tierra sobre Sol ~9 cm/s
    106: "VERDADERO",       # Probabilidad detección planeta tipo Tierra por tránsito ~0,5%
    107: "FALSO",           # Solo "siete sistemas" con exoplanetas — en realidad hay miles
    108: "VERDADERO",       # Einstein reconoció que espacio y tiempo no son absolutos
    109: "VERDADERO",       # Relatividad general: gravedad = curvatura del espacio-tiempo
    110: "VERDADERO",       # Fuerza gravitatoria entre electrones < fuerza eléctrica entre ellos
    111: "VERDADERO",       # Einstein desarrolló teoría general de la relatividad
    112: "VERDADERO",       # Teoría cuántica de campos unificó electromagnetismo e interacciones nucleares
    113: "VERDADERO",       # Campos cuánticos limitados por la velocidad de la luz
    114: "VERDADERO",       # Relatividad general explica fenómenos a velocidades altas y escalas grandes
    115: "VERDADERO",       # Argumento teórico: superposición cuántica implicaría curvatura en superposición

    # ── POLÍTICA (116-153) ────────────────────────────────────────────────────
    116: "NO_VERIFICABLE",  # WTFAB gestionada por hijas expresidente — detalles muy específicos caso
    117: "NO_VERIFICABLE",  # Audiencia Nacional señaló WTFAB Caso Plus Ultra — específico
    118: "VERDADERO",       # Juez Calama investiga tráfico influencias, falsedad, blanqueo; rescate 53M€ Plus Ultra
    119: "NO_VERIFICABLE",  # WTFAB recibió >1M€ 2020-2025 procedente de trama — muy específico
    120: "NO_VERIFICABLE",  # UCO registró sede WTFAB — detalle procesal específico
    121: "NO_VERIFICABLE",  # Pagos WTFAB canalizados mediante contratos consultoría — específico
    122: "VERDADERO",       # Donald Trump llegó a la presidencia de EEUU
    123: "VERDADERO",       # Turquía es miembro de la OTAN
    124: "VERDADERO",       # India tiene disputas fronterizas con China
    125: "VERDADERO",       # UE proporcionó apoyo a Ucrania durante conflicto con Rusia
    126: "VERDADERO",       # Eslovaquia bajo Robert Fico es miembro de UE y OTAN
    127: "VERDADERO",       # EEUU e Israel mantienen tensiones/conflicto con Irán en Oriente Medio
    128: "VERDADERO",       # Irán coopera militarmente con Rusia (suministró drones)
    129: "VERDADERO",       # Existe comunidad de judíos de origen ruso en Israel (inmigración años 90)
    130: "VERDADERO",       # Estado de Israel creado en 1948
    131: "VERDADERO",       # Operación Paz para Galilea 1982, invasión sur del Líbano
    132: "VERDADERO",       # Liga árabe atacó Israel 1948 (Egipto, Jordania, Siria, Irak, Líbano)
    133: "VERDADERO",       # OLP liderada por Arafat operó desde el Líbano
    134: "VERDADERO",       # Revolución Islámica Irán (1979) pocos años antes de invasión israelí Líbano (1982)
    135: "VERDADERO",       # Israel se retiró de Beirut en 1983
    136: "VERDADERO",       # Israel retirada unilateral del Líbano en el año 2000
    137: "VERDADERO",       # Guerra árabe-israelí 1948 concluyó con consolidación territorio israelí
    138: "VERDADERO",       # Lenin nació en 1870 en Simbirsk (Sinbirsk = error transcripción)
    139: "VERDADERO",       # Alexander Ilyich Ulyanov arrestado y ejecutado en 1887
    140: "VERDADERO",       # Lenin escribió "¿Qué hacer?" en 1902
    141: "VERDADERO",       # Derrota de Rusia frente a Japón ocurrió en 1905
    142: "VERDADERO",       # Primera Guerra Mundial estalló en 1914
    143: "VERDADERO",       # Alexander Ulyanov planeó asesinar al zar Alejandro III
    144: "VERDADERO",       # Lenin conoció a Nadezhda Krupskaya, su esposa
    145: "VERDADERO",       # Lenin y Martov fundaron Liga de lucha por la emancipación en 1895
    146: "NO_VERIFICABLE",  # 153 documentos bajo protección oficial — número muy específico
    147: "NO_VERIFICABLE",  # Esquema pintado noviembre 1980 — detalle muy específico
    148: "VERDADERO",       # Milans del Bosch sacó tanques a las calles de Valencia el 23F 1981
    149: "NO_VERIFICABLE",  # Publicación 153 documentos clasificados 23F — reciente
    150: "NO_VERIFICABLE",  # Armada mencionó conversaciones con Juan Carlos I en juicio — específico
    151: "VERDADERO",       # 45 años desde el 23F (grabado en 2026: 1981+45=2026)
    152: "NO_VERIFICABLE",  # Equipo Diario.es 24h explorando documentos — específico
    153: "NO_VERIFICABLE",  # Sin indicios que impliquen al rey en 23F — específico/reciente

    # ── HISTORIA (154-193) ────────────────────────────────────────────────────
    154: "VERDADERO",       # Octavio Augusto gobernó Imperio Romano 27 aC - 14 dC
    155: "VERDADERO",       # Conexiones comerciales Roma-China a través de la Ruta de la Seda
    156: "VERDADERO",       # Anarquía Militar en el siglo III del Imperio Romano
    157: "FALSO",           # República Romana NO terminó en 14 aC — terminó en 27 aC (inicio de Augusto); 14 dC fue su muerte
    158: "VERDADERO",       # Augusto participó en las guerras Cántabras en España
    159: "VERDADERO",       # Hubo contactos entre Imperio Romano (Augusto) y China de la Dinastía Han
    160: "VERDADERO",       # Marco Aurelio murió ~180 dC; tras su muerte el Imperio entró en crisis
    161: "VERDADERO",       # Diocleciano llegó al poder en el Imperio Romano hacia el año 284 dC
    162: "VERDADERO",       # Revolución Francesa entre 5 mayo 1789 y 9 noviembre 1799
    163: "VERDADERO",       # Burguesía y pueblo tomaron la Bastilla el 14 de julio de 1789
    164: "VERDADERO",       # Fiesta Nacional de Francia el 14 de julio conmemora la toma de la Bastilla
    165: "VERDADERO",       # Convención Nacional francesa votó la ejecución de Luis XVI
    166: "VERDADERO",       # Revolución Francesa resultó en el fin del absolutismo monárquico
    167: "VERDADERO",       # El Tercer Estado presentó propuestas de limitación del poder real
    168: "FALSO",           # La PRIMERA sesión de los Estados Generales fue el 5 de mayo de 1789, no el 17 de junio (ese día el Tercer Estado se declaró Asamblea Nacional)
    169: "VERDADERO",       # 4 agosto 1789 la Asamblea abolió el diezmo y el feudalismo
    170: "VERDADERO",       # Thor, Odin y Loki son dioses de la mitología nórdica
    171: "VERDADERO",       # Mitología nórdica: creencias pueblos germanos/escandinavos pre-cristианismo
    172: "VERDADERO",       # Snorri Sturluson escribió la Edda Prosaica (~1220)
    173: "VERDADERO",       # Mitos nórdicos conservados por escrito en Islandia siglos XIII-XIV
    174: "VERDADERO",       # Edda Poética y Edda Prosa son las principales fuentes de mitología nórdica
    175: "VERDADERO",       # Freyr: dios nórdico de fertilidad, abundancia y paz
    176: "VERDADERO",       # Freyja: diosa nórdica del amor, belleza, magia y guerra
    177: "VERDADERO",       # Njord (Nord) era el dios nórdico del mar y la prosperidad
    178: "VERDADERO",       # Robert Koldewey comenzó excavaciones en Babilonia el 26 de marzo de 1899
    179: "VERDADERO",       # Batalla de Iwo Jima fue una de las más sangrientas de la WWII en 1945
    180: "VERDADERO",       # Koldewey empleó 18 años excavando Babilonia (1899-1917)
    181: "NO_VERIFICABLE",  # 8 km + 18 km adicionales de muralla — cifras específicas no verificadas
    182: "NO_VERIFICABLE",  # Babilonia 80.000 habitantes — estimaciones varían mucho
    183: "NO_VERIFICABLE",  # Primer muro de ladrillo el 5 de abril, 11 días después — muy específico
    184: "VERDADERO",       # Puerta de Istar (Ishtar Gate) se encuentra en el Museo Pérgamo de Berlín
    185: "VERDADERO",       # Puerta de Istar construida hace ~2.600 años (~575 aC)
    186: "VERDADERO",       # Autoridades francesas reclamaron restos de Napoleón en 1840
    187: "VERDADERO",       # Napoleón murió en 1821, traslado en 1840: ~19 años después
    188: "VERDADERO",       # 12 mayo 1840 Asamblea informada del permiso británico para trasladar restos
    189: "VERDADERO",       # Fragata partió Tolón julio 1840, llegó Santa Elena octubre 1840
    190: "FALSO",           # La expedición tardó 3 meses (julio a octubre), NO 4 meses como dice la claim
    191: "VERDADERO",       # Restos de Napoleón reconocibles ~20 años después de su muerte (1821→1840)
    192: "NO_VERIFICABLE",  # 1 millón de francos aprobados — cifra específica no verificada
    193: "FALSO",           # El barco era la "Belle Poule", NO la "San Fermín" (error de transcripción Whisper)
}

# Cargar CSV con separador explícito
df = pd.read_csv(CSV, encoding="utf-8", sep=",", engine="python", dtype=str)
print(f"Claims cargadas: {len(df)}")
assert len(df) == len(LABELS), f"Número de claims ({len(df)}) != número de etiquetas ({len(LABELS)})"

# Restaurar orden original: por URL (según VIDEO_ORDER) y dentro de cada vídeo por claim_score desc
url_rank = {url: i for i, url in enumerate(VIDEO_ORDER)}
df["_url_rank"] = df["video_url"].map(url_rank)
df["_score_num"] = pd.to_numeric(df["claim_score"], errors="coerce")
df = df.sort_values(["_url_rank", "_score_num"], ascending=[True, False]).reset_index(drop=True)
df = df.drop(columns=["_url_rank", "_score_num"])

# Aplicar etiquetas en el orden restaurado
for i in range(len(df)):
    df.at[i, "true_label"] = LABELS[i + 1]

# Guardar con separador coma explícito
df.to_csv(CSV, index=False, encoding="utf-8", sep=",")
print(f"Etiquetas aplicadas y guardadas en {CSV.name}")

# Estadísticas
from collections import Counter
counts = Counter(LABELS.values())
total = len(LABELS)
print(f"\nDistribución de etiquetas:")
for label, cnt in sorted(counts.items()):
    print(f"  {label}: {cnt} ({cnt/total*100:.1f}%)")
