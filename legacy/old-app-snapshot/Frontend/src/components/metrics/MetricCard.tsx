'use client';

import type { ReactNode } from 'react';
import { clsx } from 'clsx';

interface MetricCardProps {
  icon: ReactNode;
  label: string;
  value: string | number | null;
  unit?: string;
  badge?: string;
  badgeVariant?: 'red' | 'blue' | 'amber' | 'orange' | 'green' | 'slate' | 'navy';
  source?: string;
  loading?: boolean;
  footnote?: string;
}

export function MetricCard({
  icon,
  label,
  value,
  unit,
  badge,
  badgeVariant = 'slate',
  source,
  loading,
  footnote,
}: MetricCardProps) {
  return (
    <div className="card card-hover flex flex-col gap-4 min-h-[180px]">
      {/* Top row: icon + label */}
      <div className="flex items-center gap-2.5">
        <div className="w-9 h-9 rounded-lg bg-bg flex items-center justify-center text-navy flex-shrink-0">
          {icon}
        </div>
        <p className="label-sm">{label}</p>
      </div>

      {/* Metric value */}
      {loading ? (
        <div className="flex-1 flex flex-col gap-2 justify-center">
          <div className="skeleton h-10 w-3/4" />
          <div className="skeleton h-4 w-1/2" />
        </div>
      ) : (
        <div className="flex-1 flex flex-col justify-center">
          <div className="flex items-end gap-1.5">
            <span className={clsx('metric-number', value === null && 'text-muted text-2xl')}>
              {value === null ? '—' : value}
            </span>
            {unit && value !== null && (
              <span className="text-sm text-text-2 mb-1 font-mono">{unit}</span>
            )}
          </div>
          {badge && (
            <span className={`badge badge-${badgeVariant} mt-2 w-fit`}>{badge}</span>
          )}
          {value === null && (
            <p className="text-xs text-muted mt-1">Data unavailable</p>
          )}
        </div>
      )}

      {/* Bottom: source */}
      <div>
        <hr className="divider my-0 mb-2" />
        {source ? (
          <p className="label-sm truncate">{source}</p>
        ) : (
          <div className="skeleton h-3 w-2/3" />
        )}
        {footnote && <p className="label-sm mt-0.5 text-muted">{footnote}</p>}
      </div>
    </div>
  );
}
