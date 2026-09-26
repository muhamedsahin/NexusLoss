import { classification } from "./classification";
import { metric, ranking } from "./embed";
import { generative, selfSupervised } from "./learn";
import { regression } from "./regression";
import { distillation, pointCloud, reinforcement, sequence, survival } from "./rest";
import type { CategoryDoc } from "./types";
import { detection, segmentation } from "./vision";

export const categories: CategoryDoc[] = [
  regression,
  classification,
  segmentation,
  detection,
  metric,
  ranking,
  generative,
  selfSupervised,
  distillation,
  sequence,
  pointCloud,
  reinforcement,
  survival,
];

export const lossCount = categories.reduce((sum, category) => sum + category.losses.length, 0);
