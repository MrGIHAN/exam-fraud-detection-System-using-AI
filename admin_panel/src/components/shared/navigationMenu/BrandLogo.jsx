export default function BrandLogo({ className = "" }) {
  return (
    <svg
      className={className}
      width="140"
      height="40"
      viewBox="0 0 140 40"
      role="img"
      aria-label="EXAM PROCTOR"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect width="140" height="40" fill="#fff" />

      {/* First line: EXAM */}
      <text
        x="8"
        y="18"
        fontFamily="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"
        fontWeight="800"
        fontSize="18"
        letterSpacing="0.5"
        fill="#111"
      >
        EXAM
      </text>

      {/* Second line: PROCTOR */}
      <text
        x="8"
        y="34"
        fontFamily="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"
        fontWeight="700"
        fontSize="12"
        letterSpacing="0.6"
        fill="#111"
      >
        PROCTOR
      </text>

      {/* Underline under EXAM */}
      <rect x="8" y="21" width="48" height="2" fill="#111" rx="1" />
    </svg>
  );
}
