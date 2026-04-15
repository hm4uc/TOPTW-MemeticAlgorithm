"""Check Solomon instances for POIs with late time windows (potential Nightlife)."""
import csv
import os

instances = ['C101', 'C201', 'R101', 'R201', 'RC101', 'RC201']
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'solomon_instances')

print('=' * 90)
print('  PHAN TICH TIME WINDOWS - TIM POIs MO BAN DEM')
print('=' * 90)

all_results = {}
for inst in instances:
    path = os.path.join(base_dir, f'{inst}.csv')
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    depot = rows[0]
    depot_due = int(depot['DUE DATE'])
    print(f'\n--- {inst} --- (Depot DUE DATE = {depot_due})')

    # Collect all POI time data
    pois_data = []
    for row in rows[1:]:
        cust = int(row['CUST NO.'])
        ready = int(row['READY TIME'])
        due = int(row['DUE DATE'])
        demand = int(row['DEMAND'])
        service = int(row['SERVICE TIME'])
        window = due - ready
        pois_data.append({
            'cust': cust, 'ready': ready, 'due': due,
            'demand': demand, 'service': service, 'window': window
        })

    # Stats
    readys = [p['ready'] for p in pois_data]
    dues = [p['due'] for p in pois_data]
    print(f'  READY TIME: min={min(readys)}, max={max(readys)}, avg={sum(readys)/len(readys):.0f}')
    print(f'  DUE DATE:   min={min(dues)}, max={max(dues)}, avg={sum(dues)/len(dues):.0f}')

    # Threshold: "late" = READY TIME in top 25% of depot's horizon
    threshold_75 = depot_due * 0.75
    threshold_50 = depot_due * 0.50

    late_pois = [p for p in pois_data if p['ready'] >= threshold_75]
    mid_late_pois = [p for p in pois_data if threshold_50 <= p['ready'] < threshold_75]

    print(f'  Depot horizon 75% = {threshold_75:.0f}, 50% = {threshold_50:.0f}')
    print(f'  POIs mo sau 75% horizon (READY >= {threshold_75:.0f}): {len(late_pois)}')
    print(f'  POIs mo sau 50% horizon (READY >= {threshold_50:.0f}): {len(mid_late_pois) + len(late_pois)}')

    if late_pois:
        print(f'  --- Top POIs mo muon nhat ---')
        for p in sorted(late_pois, key=lambda x: x['ready'], reverse=True)[:10]:
            print(f"    CUST={p['cust']:>3}, READY={p['ready']:>5}, DUE={p['due']:>5}, "
                  f"DEMAND={p['demand']:>3}, WINDOW={p['window']:>4}")

    all_results[inst] = {
        'total': len(pois_data),
        'late_75': len(late_pois),
        'late_50': len(mid_late_pois) + len(late_pois),
        'depot_due': depot_due,
    }

print('\n' + '=' * 90)
print('  TONG KET')
print('=' * 90)
for inst, r in all_results.items():
    pct_75 = r['late_75'] / r['total'] * 100
    pct_50 = r['late_50'] / r['total'] * 100
    print(f"  {inst}: {r['total']} POIs | "
          f"Mo muon (>=75%): {r['late_75']:>3} ({pct_75:>5.1f}%) | "
          f"Mo muon (>=50%): {r['late_50']:>3} ({pct_50:>5.1f}%)")
