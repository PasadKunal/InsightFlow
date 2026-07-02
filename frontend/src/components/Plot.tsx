import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

// A thin React wrapper around Plotly. The backend hands us a full figure
// ({ data, layout }); we just render it responsively with no mode bar, so the
// charts feel like part of the UI rather than an embedded tool.
export function Plot({ figure, height = 320 }: { figure: any; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !figure) return;
    Plotly.react(
      el,
      figure.data,
      { ...figure.layout, autosize: true, height, margin: { l: 55, r: 20, t: 50, b: 45 } },
      { responsive: true, displayModeBar: false },
    );
  }, [figure, height]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} className="w-full" style={{ height }} />;
}
