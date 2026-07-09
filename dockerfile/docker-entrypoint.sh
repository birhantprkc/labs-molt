#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

USER=${LOCAL_USER:-"root"}

if [[ "${USER}" != "root" ]]; then
    USER_ID=${LOCAL_USER_ID:-9001}
    echo ${USER}
    echo ${USER_ID}

    chown ${USER_ID} /home/${USER}
    useradd --shell /bin/bash -u ${USER_ID} -o -c "" -m ${USER}
    usermod -a -G root ${USER}
    adduser ${USER} sudo

    # user:password
    echo "${USER}:123" | chpasswd

    export HOME=/home/${USER}
    export PATH=/home/${USER}/.local/bin/:$PATH
else
    export PATH=/root/.local/bin/:$PATH
fi

cd $HOME
exec gosu ${USER} "$@"