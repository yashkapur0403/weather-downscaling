import Link from 'next/link';

interface BrandProps {
  href?: string;
}

/** Obsidian wordmark — shared by the landing and dashboard navbars. */
export function Brand({ href = '/' }: BrandProps) {
  return (
    <Link href={href} className="flex items-center gap-2.5 no-underline">
      <img
        src="/logo.png"
        alt="Obsidian logo"
        style={{
          height: '2rem',
          width: 'auto',
          display: 'block',
          flexShrink: 0,
        }}
      />
      <div>
        <p className="font-bold text-xs leading-none" style={{ color: 'var(--text)' }}>
          Obsidian
        </p>
        <p className="text-[0.5rem] tracking-widest uppercase" style={{ color: 'var(--muted)' }}>
          Crop Intelligence
        </p>
      </div>
    </Link>
  );
}
